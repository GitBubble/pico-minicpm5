/* SPDX-License-Identifier: Apache-2.0
 *
 * Persistent multi-model SVP_ACL executor for split PICO graphs.
 *
 * The process owns one ACL runtime/device session and loads every --model OM
 * exactly once.  Callers exchange byte-exact tensor payloads over stdin/stdout
 * using the little-endian protocol documented below.  Tensor counts and sizes
 * are obtained from each loaded model descriptor and are checked on every
 * execute request.  No tensor role inference, dtype conversion, or fallback
 * computation happens here.
 *
 * Ready frame (executor -> caller):
 *   u32 "PXR1", u16 version, u16 status, u32 model_count, u32 error_bytes
 *   status == 0: for each model: u32 inputs, u32 outputs, then u64 sizes[]
 *   status != 0: error_bytes of UTF-8 diagnostic text
 *
 * Execute request (caller -> executor):
 *   u32 "PEQ1", u16 version, u16 opcode(1=execute,2=shutdown),
 *   u32 model_index, u32 input_count, u32 output_count, u32 reserved,
 *   u64 input_sizes[], u64 expected_output_sizes[], input payloads[]
 *
 * Resident-input ops (3 and 4).  Every model's input dataset is allocated once
 * at load time and its device buffers already survive across executes -- opcode
 * 1 simply overwrites all of them every call.  For decode that is the dominant
 * cost: the KV window handed to each layer grows by one row per token, so
 * re-sending the whole window spends ~34% of the token budget pushing bytes
 * that did not change through a pipe.  These two ops let a caller update only
 * the bytes that moved and then execute against the retained buffers.
 *
 * Write-input request (caller -> executor):
 *   u32 "PEQ1", u16 version, u16 opcode(3), u32 model_index,
 *   u32 input_index, u32 zero(unused), u32 reserved,
 *   u64 offset, u64 length, then length payload bytes
 *   -> written at input[input_index] + offset; responds status-only.
 *
 * Resident execute request (caller -> executor):
 *   u32 "PEQ1", u16 version, u16 opcode(4), u32 model_index,
 *   u32 public_input_prefix, u32 output_count, u32 write_count,
 *   u64 expected_output_sizes[],
 *   then write_count records of: u32 input_index, u32 flags, u64 offset,
 *   u64 length, followed by
 *     flags == 0 (WRITE_FLAG_PAYLOAD): length payload bytes from the pipe;
 *     flags == 1 (WRITE_FLAG_CHAIN):   u32 src_model, u32 src_output, u64 pad,
 *       and the length bytes are copied device-to-device out of that model's
 *       output buffer instead of crossing the pipe at all.
 *
 * The chain form exists because a split decode schedule hands each layer's
 * hidden state to the next one: without it the host must read the output back
 * and write it forward, which is two pipe crossings per layer boundary for
 * bytes that never needed to leave the device.  Both buffers are ordinary host
 * pointers into device memory inside this process, so the copy is a memcpy.
 *   -> buffers [0, public_input_prefix) keep whatever the embedded writes (or
 *   opcode 3, or a previous execute) left in them, and the descriptor tail is
 *   re-zeroed exactly as opcode 1 does, so the numeric contract is unchanged.
 *
 * The embedded write list exists because opcode 3 costs a synchronous
 * round-trip each: measured on the deployed schedule, the 24 shared_R segments
 * push only 0.27 MB yet spend 12.3 ms across 120 separate write calls.  Folding
 * the writes into the execute frame makes it one request and one response per
 * segment.  write_count == 0 is the plain form.
 *
 * Argmax request (caller -> executor):
 *   u32 "PEQ1", u16 version, u16 opcode(5), u32 model_index,
 *   u32 output_index, u32 zero, u32 zero
 *   -> responds with one 8-byte tensor: u32 index, f32 value, taken over the
 *   named FP32 output without copying it anywhere.
 *
 * A decode head publishes vocab-sized logits purely so the host can take an
 * argmax of them.  Shipping 522 KB across the pipe to extract 4 bytes costs
 * 2.4 ms per token here, and the board's Python has no numpy, so the host-side
 * reduction is worse still.  The buffer is already mapped in this process.
 *
 * Resident scatter request (opcode 6):
 *   u32 "PEQ1", u16 version, u16 opcode(6), u32 destination_model,
 *   u32 record_count, u32 zero, u32 zero,
 *   followed by record_count fixed 48-byte records:
 *     u32 destination_input, u32 source_model, u32 source_output, u32 flags,
 *     u64 destination_base, u64 destination_channel_stride,
 *     u32 channels, u32 elements_per_channel, u64 reserved
 *   -> converts a contiguous channel-major FP32 source output to IEEE FP16
 *      round-to-nearest-even and scatters each row into a resident input.
 *
 * This removes the last host round trip in packed-K/V decode graphs: current
 * rows stay in the executor, while the generic record carries all layout
 * information.  No model-specific dimensions or tensor roles live here.
 *
 * Opcodes 1 and 2 are untouched and stay wire-compatible, so an older caller
 * keeps working against this binary byte for byte.
 *
 * Execute response (executor -> caller):
 *   u32 "PES1", u16 version, u16 status, u32 model_index,
 *   u32 output_count, u32 error_bytes, u32 reserved,
 *   status == 0: u64 output_sizes[], output payloads[]
 *   status != 0: error_bytes of UTF-8 diagnostic text
 *
 * This binary deliberately does not implement MiniCPM semantics.  The split
 * runner above it binds the compiler-qualified A(q,k,v),
 * R(current_v,q_rope,current_k), B, C and head-v4 physical contracts.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define PROTOCOL_VERSION 1u
#define READY_MAGIC UINT32_C(0x31525850)    /* PXR1 */
#define REQUEST_MAGIC UINT32_C(0x31514550)  /* PEQ1 */
#define RESPONSE_MAGIC UINT32_C(0x31534550) /* PES1 */
#define OP_EXECUTE 1u
#define OP_SHUTDOWN 2u
#define OP_WRITE_INPUT 3u
#define OP_EXECUTE_RESIDENT 4u
#define OP_ARGMAX 5u
#define OP_SCATTER_F32_TO_F16 6u
#define WRITE_FLAG_PAYLOAD 0u
#define WRITE_FLAG_CHAIN 1u
#define MAX_MODELS 128u
#define MAX_MODEL_IO 64u
#define MAX_ERROR_BYTES 4096u
#define MAX_TENSOR_BYTES (UINT64_C(1) << 34)

#ifndef PICO_PERSISTENT_ACL_CONTRACT_ONLY
typedef struct {
    const char *model_paths[MAX_MODELS];
    size_t model_count;
    const char *acl_config;
    int device_id;
    int cached;
} executor_options;
#endif

static void put_u16(unsigned char *out, uint16_t value)
{
    out[0] = (unsigned char)(value & UINT16_C(0xff));
    out[1] = (unsigned char)(value >> 8);
}

static void put_u32(unsigned char *out, uint32_t value)
{
    size_t i;
    for (i = 0; i < 4; i++) out[i] = (unsigned char)(value >> (8u * i));
}

static void put_u64(unsigned char *out, uint64_t value)
{
    size_t i;
    for (i = 0; i < 8; i++) out[i] = (unsigned char)(value >> (8u * i));
}

static uint16_t get_u16(const unsigned char *in)
{
    return (uint16_t)((uint16_t)in[0] | ((uint16_t)in[1] << 8));
}

static uint32_t get_u32(const unsigned char *in)
{
    return (uint32_t)in[0] | ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) | ((uint32_t)in[3] << 24);
}

static uint64_t get_u64(const unsigned char *in)
{
    uint64_t value = 0;
    size_t i;
    for (i = 0; i < 8; i++) value |= (uint64_t)in[i] << (8u * i);
    return value;
}

static uint16_t fp32_to_fp16_rne(float value)
{
    uint32_t bits;
    uint32_t sign;
    uint32_t exponent;
    uint32_t mantissa;
    int32_t half_exponent;
    uint32_t half_mantissa;
    uint32_t remainder;
    memcpy(&bits, &value, sizeof(bits));
    sign = (bits >> 16) & UINT32_C(0x8000);
    exponent = (bits >> 23) & UINT32_C(0xff);
    mantissa = bits & UINT32_C(0x7fffff);
    if (exponent == UINT32_C(0xff)) {
        if (mantissa == 0) return (uint16_t)(sign | UINT32_C(0x7c00));
        return (uint16_t)(sign | UINT32_C(0x7e00));
    }
    half_exponent = (int32_t)exponent - 127 + 15;
    if (half_exponent >= 31)
        return (uint16_t)(sign | UINT32_C(0x7c00));
    if (half_exponent <= 0) {
        uint32_t shift;
        uint32_t halfway;
        if (half_exponent < -10) return (uint16_t)sign;
        mantissa |= UINT32_C(0x800000);
        shift = (uint32_t)(14 - half_exponent);
        half_mantissa = mantissa >> shift;
        remainder = mantissa & ((UINT32_C(1) << shift) - 1u);
        halfway = UINT32_C(1) << (shift - 1u);
        if (remainder > halfway ||
            (remainder == halfway && (half_mantissa & 1u) != 0)) {
            half_mantissa++;
        }
        return (uint16_t)(sign | half_mantissa);
    }
    half_mantissa = mantissa >> 13;
    remainder = mantissa & UINT32_C(0x1fff);
    if (remainder > UINT32_C(0x1000) ||
        (remainder == UINT32_C(0x1000) && (half_mantissa & 1u) != 0)) {
        half_mantissa++;
        if (half_mantissa == UINT32_C(0x400)) {
            half_mantissa = 0;
            half_exponent++;
            if (half_exponent >= 31)
                return (uint16_t)(sign | UINT32_C(0x7c00));
        }
    }
    return (uint16_t)(sign | ((uint32_t)half_exponent << 10) |
                      half_mantissa);
}

#ifndef PICO_PERSISTENT_ACL_CONTRACT_ONLY

static int read_exact_fd(int fd, void *buffer, size_t size)
{
    size_t done = 0;
    while (done < size) {
        ssize_t result = read(fd, (unsigned char *)buffer + done, size - done);
        if (result == 0) return done == 0 ? 1 : -1;
        if (result < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        done += (size_t)result;
    }
    return 0;
}

static int write_exact_fd(int fd, const void *buffer, size_t size)
{
    size_t done = 0;
    while (done < size) {
        ssize_t result = write(fd, (const unsigned char *)buffer + done,
                               size - done);
        if (result < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (result == 0) return -1;
        done += (size_t)result;
    }
    return 0;
}

static int write_u64_fd(int fd, uint64_t value)
{
    unsigned char bytes[8];
    put_u64(bytes, value);
    return write_exact_fd(fd, bytes, sizeof(bytes));
}

static int read_u64_fd(int fd, uint64_t *value)
{
    unsigned char bytes[8];
    int result = read_exact_fd(fd, bytes, sizeof(bytes));
    if (result != 0) return result;
    *value = get_u64(bytes);
    return 0;
}

static void usage(const char *program)
{
    fprintf(stderr,
            "Usage: %s --model MODEL.om [--model MODEL.om ...] "
            "[--device N] [--acl-config PATH] [--no-cache]\n",
            program);
}

static int parse_nonnegative_int(const char *text, int *value)
{
    char *end = NULL;
    long parsed;
    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < 0 ||
        parsed > INT32_MAX) {
        return -1;
    }
    *value = (int)parsed;
    return 0;
}

static int parse_args(int argc, char **argv, executor_options *options)
{
    int index;
    memset(options, 0, sizeof(*options));
    options->cached = 1;
    for (index = 1; index < argc; index++) {
        if (strcmp(argv[index], "--model") == 0 && index + 1 < argc) {
            if (options->model_count >= MAX_MODELS) return -1;
            options->model_paths[options->model_count++] = argv[++index];
        } else if (strcmp(argv[index], "--device") == 0 &&
                   index + 1 < argc) {
            if (parse_nonnegative_int(argv[++index],
                                      &options->device_id) != 0) return -1;
        } else if (strcmp(argv[index], "--acl-config") == 0 &&
                   index + 1 < argc) {
            options->acl_config = argv[++index];
        } else if (strcmp(argv[index], "--no-cache") == 0) {
            options->cached = 0;
        } else if (strcmp(argv[index], "--help") == 0 ||
                   strcmp(argv[index], "-h") == 0) {
            usage(argv[0]);
            exit(0);
        } else {
            return -1;
        }
    }
    return options->model_count == 0 ? -1 : 0;
}

#endif

#ifdef PICO_PERSISTENT_ACL_CONTRACT_ONLY

int main(void)
{
    unsigned char bytes[8];
    const uint64_t probe = UINT64_C(0xfedcba9876543210);
    put_u64(bytes, probe);
    if (get_u64(bytes) != probe) return 1;
    put_u32(bytes, REQUEST_MAGIC);
    if (get_u32(bytes) != REQUEST_MAGIC) return 1;
    put_u16(bytes, PROTOCOL_VERSION);
    if (get_u16(bytes) != PROTOCOL_VERSION) return 1;
    if (fp32_to_fp16_rne(1.0f) != UINT16_C(0x3c00) ||
        fp32_to_fp16_rne(-2.0f) != UINT16_C(0xc000) ||
        fp32_to_fp16_rne(0.0f) != UINT16_C(0x0000) ||
        fp32_to_fp16_rne(65504.0f) != UINT16_C(0x7bff) ||
        /* Midpoint ties: even mantissa stays, odd mantissa advances. */
        fp32_to_fp16_rne(0x1.002p0f) != UINT16_C(0x3c00) ||
        fp32_to_fp16_rne(0x1.006p0f) != UINT16_C(0x3c02) ||
        fp32_to_fp16_rne(0x1p-24f) != UINT16_C(0x0001) ||
        fp32_to_fp16_rne(0x1p-25f) != UINT16_C(0x0000)) return 1;
    printf("{\"schema\":\"pico.persistent_acl_executor.protocol.v1\","
           "\"byte_order\":\"little\",\"model_limit\":%u,"
           "\"io_limit\":%u,\"resident_scatter_f32_to_f16\":true,"
           "\"self_test\":true,"
           "\"model_execution\":false}\n",
           MAX_MODELS, MAX_MODEL_IO);
    return 0;
}

#else

#include "svp_acl.h"
#include "svp_acl_ext.h"
#include "svp_acl_mdl.h"
#include "svp_acl_rt.h"

/* ACL/libinstsim may write diagnostics to stdout.  Preserve a dedicated copy
 * for protocol bytes, then redirect process stdout to stderr before runtime
 * initialization so library text can never corrupt a binary frame. */
static int protocol_output_fd = STDOUT_FILENO;

typedef struct {
    const char *path;
    void *om_memory;
    size_t om_size;
    uint32_t model_id;
    int loaded;
    svp_acl_mdl_desc *desc;
    svp_acl_mdl_dataset *inputs;
    svp_acl_mdl_dataset *outputs;
    size_t input_count;
    size_t output_count;
    size_t input_sizes[MAX_MODEL_IO];
    size_t output_sizes[MAX_MODEL_IO];
} persistent_model;

static svp_acl_error malloc_svp(void **pointer, size_t size, int cached)
{
    svp_acl_error result;
    *pointer = NULL;
    result = cached
        ? svp_acl_rt_malloc_cached(pointer, size,
                                   SVP_ACL_MEM_MALLOC_NORMAL_ONLY)
        : svp_acl_rt_malloc(pointer, size, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
    if (result == SVP_ACL_SUCCESS && *pointer != NULL) {
        memset(*pointer, 0, size);
        if (cached) result = svp_acl_rt_mem_flush(*pointer, size);
        if (result != SVP_ACL_SUCCESS) {
            svp_acl_rt_free(*pointer);
            *pointer = NULL;
        }
    }
    return result;
}

static int pread_exact_fd(int fd, void *buffer, size_t size, off_t offset)
{
    size_t done = 0;
    while (done < size) {
        ssize_t result = pread(fd, (unsigned char *)buffer + done, size - done,
                               offset + (off_t)done);
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) return -1;
        done += (size_t)result;
    }
    return 0;
}

static int load_om_source(const char *path, int cached, void **memory,
                          size_t *size)
{
    struct stat status;
    int fd = -1;
    svp_acl_error result;
    *memory = NULL;
    *size = 0;
    if (stat(path, &status) != 0 || status.st_size <= 0 ||
        (uintmax_t)status.st_size > SIZE_MAX) {
        fprintf(stderr, "persistent executor: invalid OM %s\n", path);
        return -1;
    }
    *size = (size_t)status.st_size;
    result = malloc_svp(memory, *size, cached);
    if (result != SVP_ACL_SUCCESS || *memory == NULL) {
        fprintf(stderr,
                "persistent executor: allocate OM failed ret=%d bytes=%zu "
                "path=%s\n", result, *size, path);
        return -1;
    }
    fd = open(path, O_RDONLY);
    if (fd < 0 || pread_exact_fd(fd, *memory, *size, 0) != 0) {
        fprintf(stderr, "persistent executor: read OM failed %s\n", path);
        if (fd >= 0) close(fd);
        svp_acl_rt_free(*memory);
        *memory = NULL;
        return -1;
    }
    close(fd);
    if (cached &&
        svp_acl_rt_mem_flush(*memory, *size) != SVP_ACL_SUCCESS) {
        fprintf(stderr, "persistent executor: flush OM failed %s\n", path);
        svp_acl_rt_free(*memory);
        *memory = NULL;
        return -1;
    }
    return 0;
}

static void destroy_dataset(svp_acl_mdl_dataset *dataset)
{
    size_t index;
    size_t count;
    if (dataset == NULL) return;
    count = svp_acl_mdl_get_dataset_num_buffers(dataset);
    for (index = 0; index < count; index++) {
        svp_acl_data_buffer *buffer =
            svp_acl_mdl_get_dataset_buffer(dataset, index);
        if (buffer != NULL) {
            void *address = svp_acl_get_data_buffer_addr(buffer);
            if (address != NULL) svp_acl_rt_free(address);
            svp_acl_destroy_data_buffer(buffer);
        }
    }
    svp_acl_mdl_destroy_dataset(dataset);
}

static svp_acl_mdl_dataset *create_dataset(
    const svp_acl_mdl_desc *desc, int input, int cached,
    size_t *sizes, size_t count)
{
    svp_acl_mdl_dataset *dataset = svp_acl_mdl_create_dataset();
    size_t index;
    if (dataset == NULL) return NULL;
    for (index = 0; index < count; index++) {
        void *memory = NULL;
        size_t size = input
            ? svp_acl_mdl_get_input_size_by_index(desc, index)
            : svp_acl_mdl_get_output_size_by_index(desc, index);
        size_t stride = input
            ? svp_acl_mdl_get_input_default_stride(desc, index)
            : svp_acl_mdl_get_output_default_stride(desc, index);
        svp_acl_data_buffer *buffer;
        svp_acl_error result;
        if (size == 0 || stride == 0) goto fail;
        sizes[index] = size;
        result = malloc_svp(&memory, size, cached);
        if (result != SVP_ACL_SUCCESS || memory == NULL) goto fail;
        buffer = svp_acl_create_data_buffer(memory, size, stride);
        if (buffer == NULL) {
            svp_acl_rt_free(memory);
            goto fail;
        }
        result = svp_acl_mdl_add_dataset_buffer(dataset, buffer);
        if (result != SVP_ACL_SUCCESS) {
            svp_acl_destroy_data_buffer(buffer);
            svp_acl_rt_free(memory);
            goto fail;
        }
    }
    return dataset;
fail:
    fprintf(stderr, "persistent executor: create %s dataset failed at %zu\n",
            input ? "input" : "output", index);
    destroy_dataset(dataset);
    return NULL;
}

static void destroy_model(persistent_model *model)
{
    destroy_dataset(model->inputs);
    destroy_dataset(model->outputs);
    model->inputs = NULL;
    model->outputs = NULL;
    if (model->desc != NULL) svp_acl_mdl_destroy_desc(model->desc);
    model->desc = NULL;
    if (model->loaded) svp_acl_mdl_unload(model->model_id);
    model->loaded = 0;
    if (model->om_memory != NULL) svp_acl_rt_free(model->om_memory);
    model->om_memory = NULL;
}

static int load_model(persistent_model *model, const char *path, int cached)
{
    svp_acl_error result;
    memset(model, 0, sizeof(*model));
    model->path = path;
    if (load_om_source(path, cached, &model->om_memory, &model->om_size) != 0)
        goto fail;
    result = svp_acl_mdl_load_from_mem(model->om_memory, model->om_size,
                                       &model->model_id);
    if (result != SVP_ACL_SUCCESS) {
        fprintf(stderr, "persistent executor: load OM failed ret=%d path=%s\n",
                result, path);
        goto fail;
    }
    model->loaded = 1;
    model->desc = svp_acl_mdl_create_desc();
    if (model->desc == NULL ||
        svp_acl_mdl_get_desc(model->desc, model->model_id) !=
            SVP_ACL_SUCCESS) {
        fprintf(stderr, "persistent executor: get descriptor failed path=%s\n",
                path);
        goto fail;
    }
    model->input_count = svp_acl_mdl_get_num_inputs(model->desc);
    model->output_count = svp_acl_mdl_get_num_outputs(model->desc);
    if (model->input_count == 0 || model->output_count == 0 ||
        model->input_count > MAX_MODEL_IO ||
        model->output_count > MAX_MODEL_IO) {
        fprintf(stderr,
                "persistent executor: descriptor count unsupported "
                "inputs=%zu outputs=%zu path=%s\n",
                model->input_count, model->output_count, path);
        goto fail;
    }
    model->inputs = create_dataset(model->desc, 1, cached,
                                   model->input_sizes, model->input_count);
    model->outputs = create_dataset(model->desc, 0, cached,
                                    model->output_sizes, model->output_count);
    if (model->inputs == NULL || model->outputs == NULL) goto fail;
    return 0;
fail:
    destroy_model(model);
    return -1;
}

static int sync_dataset(svp_acl_mdl_dataset *dataset, int invalidate)
{
    size_t index;
    if (dataset == NULL) return -1;
    size_t count = svp_acl_mdl_get_dataset_num_buffers(dataset);
    for (index = 0; index < count; index++) {
        svp_acl_data_buffer *buffer =
            svp_acl_mdl_get_dataset_buffer(dataset, index);
        if (buffer == NULL) return -1;
        void *address = svp_acl_get_data_buffer_addr(buffer);
        size_t size = svp_acl_get_data_buffer_size(buffer);
        if (address == NULL || size == 0) return -1;
        svp_acl_error result = invalidate
            ? svp_acl_rt_mem_invalidate(address, size)
            : svp_acl_rt_mem_flush(address, size);
        if (result != SVP_ACL_SUCCESS) return -1;
    }
    return 0;
}

/* The one-shot probe starts every omitted task/work input and every output
 * buffer from zero.  Some PICO containers expose scratch tensors as descriptor
 * IO and some Reportop outputs leave physical padding untouched.  Re-establish
 * that baseline before every persistent execute so prior invocations cannot
 * leak stale workspace or padding into the next segment. */
static int zero_dataset_range(svp_acl_mdl_dataset *dataset, size_t begin,
                              int flush_cached)
{
    size_t index;
    size_t count;
    if (dataset == NULL) return -1;
    count = svp_acl_mdl_get_dataset_num_buffers(dataset);
    if (begin > count) return -1;
    for (index = begin; index < count; index++) {
        svp_acl_data_buffer *buffer =
            svp_acl_mdl_get_dataset_buffer(dataset, index);
        void *address;
        size_t size;
        if (buffer == NULL) return -1;
        address = svp_acl_get_data_buffer_addr(buffer);
        size = svp_acl_get_data_buffer_size(buffer);
        if (address == NULL || size == 0) return -1;
        memset(address, 0, size);
        if (flush_cached &&
            svp_acl_rt_mem_flush(address, size) != SVP_ACL_SUCCESS) {
            return -1;
        }
    }
    return 0;
}

static int write_ready(const persistent_model *models, size_t model_count,
                       uint16_t status, const char *error)
{
    unsigned char header[16];
    size_t model_index;
    size_t error_size = error == NULL ? 0 : strlen(error);
    if (error_size > MAX_ERROR_BYTES) error_size = MAX_ERROR_BYTES;
    put_u32(header, READY_MAGIC);
    put_u16(header + 4, PROTOCOL_VERSION);
    put_u16(header + 6, status);
    put_u32(header + 8, status == 0 ? (uint32_t)model_count : 0u);
    put_u32(header + 12, (uint32_t)error_size);
    if (write_exact_fd(protocol_output_fd, header, sizeof(header)) != 0) return -1;
    if (status != 0)
        return write_exact_fd(protocol_output_fd, error, error_size);
    for (model_index = 0; model_index < model_count; model_index++) {
        unsigned char counts[8];
        size_t index;
        put_u32(counts, (uint32_t)models[model_index].input_count);
        put_u32(counts + 4, (uint32_t)models[model_index].output_count);
        if (write_exact_fd(protocol_output_fd, counts, sizeof(counts)) != 0)
            return -1;
        for (index = 0; index < models[model_index].input_count; index++)
            if (write_u64_fd(protocol_output_fd,
                             models[model_index].input_sizes[index]) != 0)
                return -1;
        for (index = 0; index < models[model_index].output_count; index++)
            if (write_u64_fd(protocol_output_fd,
                             models[model_index].output_sizes[index]) != 0)
                return -1;
    }
    return 0;
}

static int write_response_header(uint16_t status, uint32_t model_index,
                                 uint32_t output_count, const char *error)
{
    unsigned char header[24];
    size_t error_size = error == NULL ? 0 : strlen(error);
    if (error_size > MAX_ERROR_BYTES) error_size = MAX_ERROR_BYTES;
    put_u32(header, RESPONSE_MAGIC);
    put_u16(header + 4, PROTOCOL_VERSION);
    put_u16(header + 6, status);
    put_u32(header + 8, model_index);
    put_u32(header + 12, status == 0 ? output_count : 0u);
    put_u32(header + 16, (uint32_t)error_size);
    put_u32(header + 20, 0);
    if (write_exact_fd(protocol_output_fd, header, sizeof(header)) != 0) return -1;
    if (status != 0)
        return write_exact_fd(protocol_output_fd, error, error_size);
    return 0;
}

static int discard_bytes(uint64_t bytes)
{
    unsigned char scratch[65536];
    while (bytes != 0) {
        size_t chunk = bytes > sizeof(scratch) ? sizeof(scratch) : (size_t)bytes;
        if (read_exact_fd(STDIN_FILENO, scratch, chunk) != 0) return -1;
        bytes -= chunk;
    }
    return 0;
}

static int serve_request(persistent_model *models, size_t model_count,
                         int cached)
{
    unsigned char header[24];
    uint64_t input_sizes[MAX_MODEL_IO] = {0};
    uint64_t output_sizes[MAX_MODEL_IO] = {0};
    uint32_t magic;
    uint16_t version;
    uint16_t opcode;
    uint32_t model_index;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t write_count;
    persistent_model *model = NULL;
    char error[MAX_ERROR_BYTES + 1] = {0};
    int valid = 1;
    size_t index;
    int result = read_exact_fd(STDIN_FILENO, header, sizeof(header));
    if (result == 1) return 1;
    if (result != 0) return -1;
    magic = get_u32(header);
    version = get_u16(header + 4);
    opcode = get_u16(header + 6);
    model_index = get_u32(header + 8);
    input_count = get_u32(header + 12);
    output_count = get_u32(header + 16);
    write_count = get_u32(header + 20);
    if (magic != REQUEST_MAGIC || version != PROTOCOL_VERSION ||
        (opcode != OP_EXECUTE_RESIDENT && write_count != 0)) {
        return -1;
    }
    if (opcode == OP_SHUTDOWN) {
        if (model_index != 0 || input_count != 0 || output_count != 0)
            return -1;
        if (write_response_header(0, 0, 0, NULL) != 0) return -1;
        return 1;
    }
    if (opcode == OP_ARGMAX) {
        unsigned char payload[8];
        uint32_t best = 0;
        float best_value = 0.0f;
        const float *values;
        size_t count;
        size_t scan;
        if (output_count != 0 || write_count != 0) return -1;
        if (model_index >= model_count ||
            input_count >= models[model_index].output_count) {
            snprintf(error, sizeof(error),
                     "argmax names model %u output %u, which does not exist",
                     model_index, input_count);
            return write_response_header(1, model_index, 0, error);
        }
        count = models[model_index].output_sizes[input_count] / sizeof(float);
        values = (const float *)svp_acl_get_data_buffer_addr(
            svp_acl_mdl_get_dataset_buffer(models[model_index].outputs,
                                           input_count));
        if (values == NULL || count == 0) {
            snprintf(error, sizeof(error),
                     "argmax output %u of model %u is empty",
                     input_count, model_index);
            return write_response_header(1, model_index, 0, error);
        }
        /* The execute path already invalidates outputs, but do it again so a
         * standalone argmax cannot read a line the NPU has superseded. */
        if (cached && svp_acl_rt_mem_invalidate(
                (void *)values,
                models[model_index].output_sizes[input_count])
                != SVP_ACL_SUCCESS) {
            snprintf(error, sizeof(error),
                     "argmax could not invalidate model %u output %u",
                     model_index, input_count);
            return write_response_header(2, model_index, 0, error);
        }
        best_value = values[0];
        for (scan = 1; scan < count; scan++) {
            if (values[scan] > best_value) {
                best_value = values[scan];
                best = (uint32_t)scan;
            }
        }
        put_u32(payload, best);
        memcpy(payload + 4, &best_value, sizeof(best_value));
        if (write_response_header(0, model_index, 1, NULL) != 0) return -1;
        if (write_u64_fd(protocol_output_fd, sizeof(payload)) != 0) return -1;
        return write_exact_fd(protocol_output_fd, payload, sizeof(payload));
    }
    if (opcode == OP_SCATTER_F32_TO_F16) {
        size_t record_index;
        if (output_count != 0 || write_count != 0 ||
            input_count == 0 || input_count > MAX_MODEL_IO) return -1;
        if (model_index >= model_count) {
            valid = 0;
            snprintf(error, sizeof(error),
                     "scatter destination model %u is out of range",
                     model_index);
        }
        for (record_index = 0; record_index < input_count; record_index++) {
            unsigned char record[48];
            uint32_t destination_input;
            uint32_t source_model;
            uint32_t source_output;
            uint32_t flags;
            uint64_t destination_base;
            uint64_t destination_stride;
            uint32_t channels;
            uint32_t elements;
            uint64_t source_elements;
            uint64_t row_bytes;
            uint64_t final_end;
            const float *source = NULL;
            unsigned char *destination = NULL;
            size_t channel;
            size_t element;
            if (read_exact_fd(STDIN_FILENO, record, sizeof(record)) != 0)
                return -1;
            destination_input = get_u32(record);
            source_model = get_u32(record + 4);
            source_output = get_u32(record + 8);
            flags = get_u32(record + 12);
            destination_base = get_u64(record + 16);
            destination_stride = get_u64(record + 24);
            channels = get_u32(record + 32);
            elements = get_u32(record + 36);
            if (flags != 0 || get_u64(record + 40) != 0 ||
                channels == 0 || elements == 0) {
                if (valid) {
                    valid = 0;
                    snprintf(error, sizeof(error),
                             "scatter record %zu has invalid flags or shape",
                             record_index);
                }
                continue;
            }
            source_elements = (uint64_t)channels * elements;
            row_bytes = (uint64_t)elements * sizeof(uint16_t);
            if (source_elements > MAX_TENSOR_BYTES / sizeof(float) ||
                (destination_base & 1u) != 0 ||
                (destination_stride & 1u) != 0 ||
                destination_stride < row_bytes ||
                destination_base > UINT64_MAX - row_bytes ||
                (uint64_t)(channels - 1u) >
                    (UINT64_MAX - destination_base - row_bytes) /
                    destination_stride) {
                if (valid) {
                    valid = 0;
                    snprintf(error, sizeof(error),
                             "scatter record %zu shape overflows",
                             record_index);
                }
                continue;
            }
            final_end = destination_base +
                (uint64_t)(channels - 1u) * destination_stride + row_bytes;
            if (!valid || model_index >= model_count ||
                destination_input >= models[model_index].input_count ||
                source_model >= model_count ||
                source_output >= models[source_model].output_count ||
                source_elements * sizeof(float) >
                    models[source_model].output_sizes[source_output] ||
                final_end > models[model_index].input_sizes[destination_input]) {
                if (valid) {
                    valid = 0;
                    snprintf(error, sizeof(error),
                             "scatter record %zu exceeds source or destination",
                             record_index);
                }
                continue;
            }
            source = (const float *)svp_acl_get_data_buffer_addr(
                svp_acl_mdl_get_dataset_buffer(models[source_model].outputs,
                                               source_output));
            destination = (unsigned char *)svp_acl_get_data_buffer_addr(
                svp_acl_mdl_get_dataset_buffer(models[model_index].inputs,
                                               destination_input));
            if (source == NULL || destination == NULL) {
                valid = 0;
                snprintf(error, sizeof(error),
                         "scatter record %zu buffer is unavailable",
                         record_index);
                continue;
            }
            if (cached && svp_acl_rt_mem_invalidate(
                    (void *)source,
                    models[source_model].output_sizes[source_output])
                    != SVP_ACL_SUCCESS) {
                valid = 0;
                snprintf(error, sizeof(error),
                         "scatter record %zu could not invalidate source",
                         record_index);
                continue;
            }
            for (channel = 0; channel < channels; channel++) {
                uint16_t *row = (uint16_t *)(destination + destination_base +
                    channel * destination_stride);
                for (element = 0; element < elements; element++) {
                    row[element] = fp32_to_fp16_rne(
                        source[channel * elements + element]);
                }
            }
        }
        if (!valid)
            return write_response_header(1, model_index, 0, error);
        return write_response_header(0, model_index, 0, NULL);
    }
    if (opcode == OP_WRITE_INPUT) {
        uint64_t offset = 0;
        uint64_t length = 0;
        if (output_count != 0) return -1;
        if (read_u64_fd(STDIN_FILENO, &offset) != 0) return -1;
        if (read_u64_fd(STDIN_FILENO, &length) != 0) return -1;
        if (length > MAX_TENSOR_BYTES) return -1;
        if (model_index >= model_count) {
            snprintf(error, sizeof(error), "model index %u is out of range",
                     model_index);
        } else if (input_count >= models[model_index].input_count) {
            snprintf(error, sizeof(error),
                     "input index %u is out of range for model %u",
                     input_count, model_index);
        } else {
            size_t capacity = models[model_index].input_sizes[input_count];
            /* Reject before reading so a bad window can never be a partial
             * write into a live device buffer. */
            if (offset > capacity || length > capacity - offset) {
                snprintf(error, sizeof(error),
                         "write [%" PRIu64 ", %" PRIu64 ") exceeds input[%u] "
                         "size %zu for model %u",
                         offset, offset + length, input_count, capacity,
                         model_index);
            } else {
                svp_acl_data_buffer *buffer = svp_acl_mdl_get_dataset_buffer(
                    models[model_index].inputs, input_count);
                unsigned char *address =
                    (unsigned char *)svp_acl_get_data_buffer_addr(buffer);
                if (address == NULL) {
                    snprintf(error, sizeof(error),
                             "input[%u] buffer is unavailable for model %u",
                             input_count, model_index);
                } else {
                    if (read_exact_fd(STDIN_FILENO, address + offset,
                                      (size_t)length) != 0) return -1;
                    return write_response_header(0, model_index, 0, NULL);
                }
            }
        }
        if (discard_bytes(length) != 0) return -1;
        return write_response_header(1, model_index, 0, error);
    }
    if ((opcode != OP_EXECUTE && opcode != OP_EXECUTE_RESIDENT) ||
        input_count > MAX_MODEL_IO || output_count > MAX_MODEL_IO) return -1;
    /* Resident execute carries no input payloads: input_count is the public
     * prefix to preserve, not a count of buffers arriving on the pipe. */
    if (opcode == OP_EXECUTE) {
        for (index = 0; index < input_count; index++) {
            if (read_u64_fd(STDIN_FILENO, &input_sizes[index]) != 0) return -1;
            if (input_sizes[index] > MAX_TENSOR_BYTES) valid = 0;
        }
    }
    for (index = 0; index < output_count; index++) {
        if (read_u64_fd(STDIN_FILENO, &output_sizes[index]) != 0) return -1;
        if (output_sizes[index] > MAX_TENSOR_BYTES) valid = 0;
    }
    if (model_index >= model_count) {
        valid = 0;
        snprintf(error, sizeof(error), "model index %u is out of range",
                 model_index);
    } else {
        model = &models[model_index];
        /* Split OMs may expose container-internal task/work tensors after the
         * public prefix.  The probe path leaves those zero-filled and reads
         * only public Reportop outputs.  Preserve that exact convention while
         * still validating every supplied prefix buffer byte-for-byte. */
        /* A chained resident execute may publish nothing: the consumer copies
         * straight out of this model's output buffer, so returning the tensor
         * over the pipe would be the exact round trip the chain removes. */
        if (input_count == 0 ||
            (output_count == 0 && opcode != OP_EXECUTE_RESIDENT) ||
            input_count > model->input_count ||
            output_count > model->output_count) {
            valid = 0;
            snprintf(error, sizeof(error),
                     "public IO prefix exceeds descriptor for model %u",
                     model_index);
        }
    }
    if (model != NULL && opcode == OP_EXECUTE &&
        input_count <= model->input_count) {
        for (index = 0; index < input_count; index++) {
            if (input_sizes[index] != model->input_sizes[index]) {
                valid = 0;
                snprintf(error, sizeof(error),
                         "input[%zu] size mismatch for model %u: "
                         "request=%" PRIu64 " descriptor=%zu",
                         index, model_index, input_sizes[index],
                         model->input_sizes[index]);
            }
        }
    }
    if (model != NULL && output_count <= model->output_count) {
        for (index = 0; index < output_count; index++) {
            if (output_sizes[index] != model->output_sizes[index]) {
                valid = 0;
                snprintf(error, sizeof(error),
                         "output[%zu] size mismatch for model %u: "
                         "request=%" PRIu64 " descriptor=%zu",
                         index, model_index, output_sizes[index],
                         model->output_sizes[index]);
            }
        }
    }
    if (opcode == OP_EXECUTE) {
        for (index = 0; index < input_count; index++) {
            if (valid) {
                svp_acl_data_buffer *buffer =
                    svp_acl_mdl_get_dataset_buffer(model->inputs, index);
                void *address = svp_acl_get_data_buffer_addr(buffer);
                if (read_exact_fd(STDIN_FILENO, address,
                                  (size_t)input_sizes[index]) != 0) return -1;
            } else if (discard_bytes(input_sizes[index]) != 0) {
                return -1;
            }
        }
    }
    /* Embedded writes are drained even when the frame is already rejected, so
     * a refused request cannot desynchronise the stream for the next one. */
    for (index = 0; index < write_count; index++) {
        unsigned char record[16];
        uint32_t slot;
        uint32_t flags;
        uint64_t offset;
        uint64_t length;
        const unsigned char *source = NULL;
        unsigned char *address = NULL;
        int writable;
        if (read_exact_fd(STDIN_FILENO, record, sizeof(record)) != 0) return -1;
        slot = get_u32(record);
        flags = get_u32(record + 4);
        offset = get_u64(record + 8);
        if (read_u64_fd(STDIN_FILENO, &length) != 0) return -1;
        if (length > MAX_TENSOR_BYTES) return -1;
        if (flags != WRITE_FLAG_PAYLOAD && flags != WRITE_FLAG_CHAIN) return -1;
        writable = valid && model != NULL && slot < model->input_count
            && offset <= model->input_sizes[slot]
            && length <= model->input_sizes[slot] - offset;
        if (writable) {
            svp_acl_data_buffer *buffer =
                svp_acl_mdl_get_dataset_buffer(model->inputs, slot);
            address = (unsigned char *)svp_acl_get_data_buffer_addr(buffer);
            if (address == NULL) writable = 0;
        }
        if (flags == WRITE_FLAG_CHAIN) {
            unsigned char link[16];
            uint32_t src_model;
            uint32_t src_output;
            if (read_exact_fd(STDIN_FILENO, link, sizeof(link)) != 0) return -1;
            src_model = get_u32(link);
            src_output = get_u32(link + 4);
            if (src_model >= model_count ||
                src_output >= models[src_model].output_count ||
                length > models[src_model].output_sizes[src_output]) {
                if (valid) {
                    valid = 0;
                    snprintf(error, sizeof(error),
                             "chain write %zu names model %u output %u, "
                             "which cannot supply %" PRIu64 " bytes",
                             index, src_model, src_output, length);
                }
                continue;
            }
            source = (const unsigned char *)svp_acl_get_data_buffer_addr(
                svp_acl_mdl_get_dataset_buffer(models[src_model].outputs,
                                               src_output));
            if (source == NULL) writable = 0;
            if (writable) {
                /* Both pointers address device memory owned by this process,
                 * so the hidden state never crosses the pipe.  Invalidate the
                 * producer's buffer first when running cached, otherwise this
                 * can read a stale line the NPU has already superseded. */
                if (cached && svp_acl_rt_mem_invalidate(
                        (void *)source,
                        models[src_model].output_sizes[src_output])
                        != SVP_ACL_SUCCESS) {
                    valid = 0;
                    snprintf(error, sizeof(error),
                             "chain write %zu could not invalidate model %u "
                             "output %u", index, src_model, src_output);
                    continue;
                }
                memcpy(address + offset, source, (size_t)length);
            }
        } else if (writable) {
            if (read_exact_fd(STDIN_FILENO, address + offset,
                              (size_t)length) != 0) return -1;
        }
        if (!writable) {
            if (valid) {
                valid = 0;
                snprintf(error, sizeof(error),
                         "embedded write %zu targets input[%u] "
                         "[%" PRIu64 ", %" PRIu64 ") outside model %u",
                         index, slot, offset, offset + length, model_index);
            }
            if (flags == WRITE_FLAG_PAYLOAD && discard_bytes(length) != 0)
                return -1;
        }
    }
    if (!valid) {
        if (error[0] == '\0')
            snprintf(error, sizeof(error), "invalid execute frame");
        return write_response_header(1, model_index, 0, error);
    }
    if (zero_dataset_range(model->inputs, input_count, 0) != 0) {
        return write_response_header(2, model_index, 0,
                                     "internal input reset failed");
    }
    if (zero_dataset_range(model->outputs, 0, cached) != 0) {
        return write_response_header(2, model_index, 0,
                                     "output reset failed");
    }
    if (cached && sync_dataset(model->inputs, 0) != 0) {
        return write_response_header(2, model_index, 0,
                                     "input cache flush failed");
    }
    {
        svp_acl_error acl_result = svp_acl_mdl_execute(
            model->model_id, model->inputs, model->outputs);
        if (acl_result != SVP_ACL_SUCCESS) {
            snprintf(error, sizeof(error),
                     "svp_acl_mdl_execute failed ret=%d model=%u",
                     acl_result, model_index);
            return write_response_header(3, model_index, 0, error);
        }
    }
    if (cached && sync_dataset(model->outputs, 1) != 0) {
        return write_response_header(4, model_index, 0,
                                     "output cache invalidate failed");
    }
    if (write_response_header(0, model_index, output_count, NULL) != 0)
        return -1;
    for (index = 0; index < output_count; index++)
        if (write_u64_fd(protocol_output_fd, model->output_sizes[index]) != 0)
            return -1;
    for (index = 0; index < output_count; index++) {
        svp_acl_data_buffer *buffer =
            svp_acl_mdl_get_dataset_buffer(model->outputs, index);
        const void *address = svp_acl_get_data_buffer_addr(buffer);
        if (write_exact_fd(protocol_output_fd, address,
                           model->output_sizes[index]) != 0) return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    executor_options options;
    persistent_model models[MAX_MODELS];
    size_t loaded_count = 0;
    svp_acl_error result;
    int acl_initialized = 0;
    int device_set = 0;
    int exit_code = 1;
    char ready_error[512] = {0};
    uintmax_t total_om_bytes = 0;
    memset(models, 0, sizeof(models));
    if (parse_args(argc, argv, &options) != 0) {
        usage(argv[0]);
        return 2;
    }
    fflush(stdout);
    protocol_output_fd = dup(STDOUT_FILENO);
    if (protocol_output_fd < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
        fprintf(stderr, "persistent executor: isolate protocol fd failed\n");
        if (protocol_output_fd >= 0) close(protocol_output_fd);
        return 2;
    }
    result = svp_acl_init(options.acl_config);
    if (result != SVP_ACL_SUCCESS) {
        snprintf(ready_error, sizeof(ready_error),
                 "svp_acl_init failed ret=%d", result);
        write_ready(NULL, 0, 1, ready_error);
        goto cleanup;
    }
    acl_initialized = 1;
    result = svp_acl_rt_set_device(options.device_id);
    if (result != SVP_ACL_SUCCESS) {
        snprintf(ready_error, sizeof(ready_error),
                 "svp_acl_rt_set_device failed ret=%d", result);
        write_ready(NULL, 0, 1, ready_error);
        goto cleanup;
    }
    device_set = 1;
    for (loaded_count = 0; loaded_count < options.model_count; loaded_count++) {
        if (load_model(&models[loaded_count],
                       options.model_paths[loaded_count],
                       options.cached) != 0) {
            snprintf(ready_error, sizeof(ready_error),
                     "load model[%zu] failed: %s", loaded_count,
                     options.model_paths[loaded_count]);
            write_ready(NULL, 0, 1, ready_error);
            goto cleanup;
        }
        fprintf(stderr,
                "persistent_executor.model[%zu]=ready id=%u inputs=%zu "
                "outputs=%zu bytes=%zu path=%s\n",
                loaded_count, models[loaded_count].model_id,
                models[loaded_count].input_count,
                models[loaded_count].output_count,
                models[loaded_count].om_size, models[loaded_count].path);
        if (UINTMAX_MAX - total_om_bytes < models[loaded_count].om_size) {
            snprintf(ready_error, sizeof(ready_error),
                     "resident OM byte total overflow at model[%zu]",
                     loaded_count);
            write_ready(NULL, 0, 1, ready_error);
            loaded_count++;
            goto cleanup;
        }
        total_om_bytes += models[loaded_count].om_size;
    }
    if (write_ready(models, options.model_count, 0, NULL) != 0) goto cleanup;
    fprintf(stderr,
            "persistent_executor=ready models=%zu cached=%d om_bytes=%" PRIuMAX
            "\n", options.model_count, options.cached, total_om_bytes);
    for (;;) {
        int serve_result = serve_request(models, options.model_count,
                                         options.cached);
        if (serve_result == 1) {
            exit_code = 0;
            break;
        }
        if (serve_result != 0) break;
    }
cleanup:
    while (loaded_count != 0) destroy_model(&models[--loaded_count]);
    if (device_set) svp_acl_rt_reset_device(options.device_id);
    if (acl_initialized) svp_acl_finalize();
    close(protocol_output_fd);
    return exit_code;
}

#endif

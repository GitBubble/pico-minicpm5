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
 * information.  No model-specific dimensions or tensor roles live here.  The
 * executor drains, validates and resolves the complete record list before it
 * touches a destination.  In cached mode every source is invalidated before
 * the first conversion, and every written destination row is flushed before a
 * success response.  A destination-flush failure poisons the native session:
 * an error response is emitted and the process exits so the caller must start
 * a fresh executor rather than consume potentially incoherent resident state.
 *
 * Resident-input snapshots (opcodes 7 and 8):
 *   save:    opcode(7), model_index, snapshot_id, range_count, zero,
 *            followed by range_count records:
 *              u32 input_index, u32 zero, u64 offset, u64 length
 *   restore: opcode(8), model_index, snapshot_id, zero, zero
 *
 * SAVE copies only the named resident-input ranges into a bounded host-memory
 * slot owned by this executor. RESTORE resolves and validates every destination
 * before the first copy, then flushes every restored cached range before ACK.
 * A restore flush failure terminates the session because earlier ranges may
 * already have changed. This supports immutable prompt-prefix K/V snapshots
 * without sending cache bytes over the pipe and without encoding any
 * model-specific layout here.
 *
 * Resident input-to-input copy (opcode 9):
 *   u32 "PEQ1", u16 version, u16 opcode(9), u32 zero,
 *   u32 record_count, u32 zero, u32 zero,
 *   followed by record_count fixed 44-byte records:
 *     u32 destination_model, u32 destination_input, u64 destination_offset,
 *     u32 source_model, u32 source_input, u64 source_offset,
 *     u64 length, u32 flags
 *
 * flags must be zero.  The executor drains and validates the complete record
 * list before touching any destination.  Cached source ranges are invalidated
 * before the copy phase and cached destination ranges are flushed afterwards.
 * Copies within one input buffer use memmove; copies between buffers use
 * memcpy.  This operation is scoped to the currently loaded model table: the
 * protocol has no model-table generation field and makes no cross-generation
 * validity claim.
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
#define OP_SNAPSHOT_INPUTS 7u
#define OP_RESTORE_INPUTS 8u
#define OP_COPY_INPUTS 9u
#define WRITE_FLAG_PAYLOAD 0u
#define WRITE_FLAG_CHAIN 1u
#define MAX_MODELS 128u
#define MAX_MODEL_IO 64u
#define SCATTER_F32_TO_F16_RECORD_BYTES 48u
#define MAX_INPUT_COPY_RECORDS 256u
#define INPUT_COPY_RECORD_BYTES 44u
#define MAX_SNAPSHOTS 8u
#define MAX_SNAPSHOT_RANGES 256u
#define MAX_SNAPSHOT_BYTES (UINT64_C(64) << 20)
#define MAX_TOTAL_SNAPSHOT_BYTES (UINT64_C(128) << 20)
#define MAX_ERROR_BYTES 4096u
#define MAX_TENSOR_BYTES (UINT64_C(1) << 34)

_Static_assert(SCATTER_F32_TO_F16_RECORD_BYTES == 48u,
               "resident FP32-to-FP16 scatter record ABI drift");
_Static_assert(INPUT_COPY_RECORD_BYTES == 44u,
               "resident input copy record ABI drift");

static uint16_t fp32_to_fp16_rne(float value);

static int scatter_f32_to_f16_bounds_valid(
    uint64_t destination_base, uint64_t destination_stride,
    uint32_t channels, uint32_t elements, uint64_t source_capacity,
    uint64_t destination_capacity, uint64_t *source_bytes_out,
    uint64_t *row_bytes_out, uint64_t *final_end_out)
{
    uint64_t source_elements;
    uint64_t source_bytes;
    uint64_t row_bytes;
    uint64_t final_end;
    if (channels == 0 || elements == 0 ||
        (uint64_t)channels >
            MAX_TENSOR_BYTES / sizeof(float) / (uint64_t)elements) {
        return 0;
    }
    source_elements = (uint64_t)channels * (uint64_t)elements;
    source_bytes = source_elements * sizeof(float);
    row_bytes = (uint64_t)elements * sizeof(uint16_t);
    if ((destination_base & 1u) != 0 ||
        (destination_stride & 1u) != 0 ||
        destination_stride < row_bytes || source_bytes > source_capacity ||
        destination_base > destination_capacity ||
        row_bytes > destination_capacity - destination_base ||
        (uint64_t)(channels - 1u) >
            (destination_capacity - destination_base - row_bytes) /
                destination_stride) {
        return 0;
    }
    final_end = destination_base +
        (uint64_t)(channels - 1u) * destination_stride + row_bytes;
    if (source_bytes_out != NULL) *source_bytes_out = source_bytes;
    if (row_bytes_out != NULL) *row_bytes_out = row_bytes;
    if (final_end_out != NULL) *final_end_out = final_end;
    return 1;
}

static void scatter_f32_to_f16_rows(
    unsigned char *destination, const float *source,
    uint64_t destination_base, uint64_t destination_stride,
    uint32_t channels, uint32_t elements)
{
    size_t channel;
    size_t element;
    for (channel = 0; channel < channels; channel++) {
        uint16_t *row = (uint16_t *)(destination + destination_base +
            channel * destination_stride);
        for (element = 0; element < elements; element++) {
            row[element] = fp32_to_fp16_rne(
                source[channel * (size_t)elements + element]);
        }
    }
}

static int input_copy_bounds_valid(uint64_t offset, uint64_t capacity,
                                   uint64_t length)
{
    return length != 0 && length <= MAX_TENSOR_BYTES &&
           offset <= capacity && length <= capacity - offset;
}

static void copy_input_bytes(unsigned char *destination,
                             const unsigned char *source, size_t length,
                             int same_buffer)
{
    if (same_buffer)
        memmove(destination, source, length);
    else
        memcpy(destination, source, length);
}

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
    unsigned char overlap[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    unsigned char atomic_destination[4] = {9, 8, 7, 6};
    const unsigned char atomic_before[4] = {9, 8, 7, 6};
    const float scatter_source[4] = {1.0f, -2.0f, 0.0f, 65504.0f};
    uint16_t scatter_destination[6] = {
        UINT16_C(0xffff), UINT16_C(0xffff), UINT16_C(0xaaaa),
        UINT16_C(0xaaaa), UINT16_C(0xffff), UINT16_C(0xffff)};
    const uint16_t scatter_expected[6] = {
        UINT16_C(0x3c00), UINT16_C(0xc000), UINT16_C(0xaaaa),
        UINT16_C(0xaaaa), UINT16_C(0x0000), UINT16_C(0x7bff)};
    uint16_t atomic_scatter_destination[4] = {
        UINT16_C(9), UINT16_C(8), UINT16_C(7), UINT16_C(6)};
    const uint16_t atomic_scatter_before[4] = {
        UINT16_C(9), UINT16_C(8), UINT16_C(7), UINT16_C(6)};
    uint64_t scatter_source_bytes = 0;
    uint64_t scatter_row_bytes = 0;
    uint64_t scatter_final_end = 0;
    int all_valid;
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
    if (!input_copy_bounds_valid(4, 8, 4) ||
        input_copy_bounds_valid(5, 8, 4) ||
        input_copy_bounds_valid(0, 8, 0) ||
        input_copy_bounds_valid(UINT64_MAX, UINT64_MAX, 2)) return 1;
    copy_input_bytes(overlap + 2, overlap, 6, 1);
    if (memcmp(overlap, (const unsigned char[]){0, 1, 0, 1, 2, 3, 4, 5},
               sizeof(overlap)) != 0) return 1;
    /* Mirror the request handler's validate-all-before-copy invariant: the
     * invalid second range prevents even the valid first range from writing. */
    all_valid = input_copy_bounds_valid(0, 4, 2) &&
                input_copy_bounds_valid(3, 4, 2);
    if (all_valid)
        copy_input_bytes(atomic_destination, bytes, 2, 0);
    if (memcmp(atomic_destination, atomic_before,
               sizeof(atomic_destination)) != 0) return 1;
    if (!scatter_f32_to_f16_bounds_valid(
            0, 8, 2, 2, sizeof(scatter_source),
            sizeof(scatter_destination), &scatter_source_bytes,
            &scatter_row_bytes, &scatter_final_end) ||
        scatter_source_bytes != sizeof(scatter_source) ||
        scatter_row_bytes != 2 * sizeof(uint16_t) ||
        scatter_final_end != sizeof(scatter_destination) ||
        scatter_f32_to_f16_bounds_valid(
            1, 8, 2, 2, sizeof(scatter_source),
            sizeof(scatter_destination), NULL, NULL, NULL) ||
        scatter_f32_to_f16_bounds_valid(
            0, 8, 2, 2, sizeof(scatter_source) - 1,
            sizeof(scatter_destination), NULL, NULL, NULL) ||
        scatter_f32_to_f16_bounds_valid(
            0, UINT64_MAX - 1u, UINT32_MAX, UINT32_MAX,
            UINT64_MAX, UINT64_MAX, NULL, NULL, NULL)) return 1;
    scatter_f32_to_f16_rows(
        (unsigned char *)scatter_destination, scatter_source, 0, 8, 2, 2);
    if (memcmp(scatter_destination, scatter_expected,
               sizeof(scatter_destination)) != 0) return 1;
    /* Mirror opcode 6's validate-all-before-convert invariant: the invalid
     * second destination suppresses the otherwise valid first record. */
    all_valid = scatter_f32_to_f16_bounds_valid(
                    0, 4, 1, 2, 2 * sizeof(float),
                    sizeof(atomic_scatter_destination), NULL, NULL, NULL) &&
                scatter_f32_to_f16_bounds_valid(
                    4, 4, 2, 2, sizeof(scatter_source),
                    sizeof(atomic_scatter_destination), NULL, NULL, NULL);
    if (all_valid) {
        scatter_f32_to_f16_rows(
            (unsigned char *)atomic_scatter_destination, scatter_source,
            0, 4, 1, 2);
    }
    if (memcmp(atomic_scatter_destination, atomic_scatter_before,
               sizeof(atomic_scatter_destination)) != 0) return 1;
    printf("{\"schema\":\"pico.persistent_acl_executor.protocol.v1\","
           "\"byte_order\":\"little\",\"model_limit\":%u,"
           "\"io_limit\":%u,\"resident_scatter_f32_to_f16\":true,"
           "\"resident_scatter_record_bytes\":%u,"
           "\"resident_scatter_validate_all\":true,"
           "\"resident_scatter_flush_failure_fatal\":true,"
           "\"resident_input_snapshots\":true,"
           "\"resident_snapshot_restore_validate_all\":true,"
           "\"resident_snapshot_restore_flush_failure_fatal\":true,"
           "\"resident_input_copy\":true,"
           "\"resident_input_copy_opcode\":%u,"
           "\"resident_input_copy_record_bytes\":%u,"
           "\"resident_input_copy_generation_guard\":false,"
           "\"self_test\":true,"
           "\"model_execution\":false}\n",
           MAX_MODELS, MAX_MODEL_IO, SCATTER_F32_TO_F16_RECORD_BYTES,
           OP_COPY_INPUTS,
           INPUT_COPY_RECORD_BYTES);
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

typedef struct {
    uint32_t input_index;
    uint64_t offset;
    uint64_t length;
    uint64_t data_offset;
} resident_snapshot_range;

typedef struct {
    int valid;
    uint32_t model_index;
    size_t range_count;
    size_t byte_count;
    resident_snapshot_range ranges[MAX_SNAPSHOT_RANGES];
    unsigned char *bytes;
} resident_snapshot;

typedef struct {
    uint32_t destination_model;
    uint32_t destination_input;
    uint64_t destination_offset;
    uint32_t source_model;
    uint32_t source_input;
    uint64_t source_offset;
    uint64_t length;
    unsigned char *destination;
    const unsigned char *source;
    int same_buffer;
} resident_input_copy;

typedef struct {
    uint32_t destination_input;
    uint32_t source_model;
    uint32_t source_output;
    uint64_t destination_base;
    uint64_t destination_stride;
    uint32_t channels;
    uint32_t elements;
    uint64_t source_bytes;
    uint64_t row_bytes;
    uint64_t final_end;
    size_t source_capacity;
    unsigned char *destination;
    const float *source;
} resident_scatter_f32_to_f16;

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

static void destroy_snapshot(resident_snapshot *snapshot)
{
    free(snapshot->bytes);
    memset(snapshot, 0, sizeof(*snapshot));
}

static int serve_request(persistent_model *models, size_t model_count,
                         int cached, resident_snapshot *snapshots)
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
    if (opcode == OP_SNAPSHOT_INPUTS) {
        resident_snapshot_range ranges[MAX_SNAPSHOT_RANGES];
        unsigned char *copy = NULL;
        uint64_t byte_count = 0;
        uint64_t total_snapshot_bytes = 0;
        size_t range_count = output_count;
        size_t range_index;
        uint32_t snapshot_id = input_count;
        if (write_count != 0 || snapshot_id == 0 ||
            snapshot_id > MAX_SNAPSHOTS || range_count == 0 ||
            range_count > MAX_SNAPSHOT_RANGES) return -1;
        memset(ranges, 0, sizeof(ranges));
        if (model_index >= model_count) {
            valid = 0;
            snprintf(error, sizeof(error),
                     "snapshot model %u is out of range", model_index);
        }
        for (range_index = 0; range_index < range_count; range_index++) {
            unsigned char record[24];
            uint32_t input_index;
            uint32_t flags;
            uint64_t offset;
            uint64_t length;
            if (read_exact_fd(STDIN_FILENO, record, sizeof(record)) != 0)
                return -1;
            input_index = get_u32(record);
            flags = get_u32(record + 4);
            offset = get_u64(record + 8);
            length = get_u64(record + 16);
            ranges[range_index].input_index = input_index;
            ranges[range_index].offset = offset;
            ranges[range_index].length = length;
            ranges[range_index].data_offset = byte_count;
            if (flags != 0 || length == 0 ||
                byte_count > MAX_SNAPSHOT_BYTES ||
                length > MAX_SNAPSHOT_BYTES - byte_count) {
                if (valid) {
                    valid = 0;
                    snprintf(error, sizeof(error),
                             "snapshot range %zu has invalid flags or size",
                             range_index);
                }
                continue;
            }
            byte_count += length;
            if (model_index >= model_count ||
                input_index >= models[model_index].input_count ||
                offset > models[model_index].input_sizes[input_index] ||
                length > models[model_index].input_sizes[input_index] -
                    offset) {
                if (valid) {
                    valid = 0;
                    snprintf(error, sizeof(error),
                             "snapshot range %zu exceeds model %u input %u",
                             range_index, model_index, input_index);
                }
            }
        }
        if (valid) {
            for (range_index = 0; range_index < MAX_SNAPSHOTS;
                 range_index++) {
                if (range_index != snapshot_id - 1u &&
                    snapshots[range_index].valid) {
                    total_snapshot_bytes += snapshots[range_index].byte_count;
                }
            }
            if (byte_count > MAX_TOTAL_SNAPSHOT_BYTES ||
                total_snapshot_bytes >
                    MAX_TOTAL_SNAPSHOT_BYTES - byte_count) {
                valid = 0;
                snprintf(error, sizeof(error),
                         "resident snapshot budget would exceed %" PRIu64
                         " bytes", MAX_TOTAL_SNAPSHOT_BYTES);
            }
        }
        if (valid) {
            copy = (unsigned char *)malloc((size_t)byte_count);
            if (copy == NULL) {
                valid = 0;
                snprintf(error, sizeof(error),
                         "could not allocate %" PRIu64
                         " bytes for snapshot %u", byte_count, snapshot_id);
            }
        }
        if (valid) {
            for (range_index = 0; range_index < range_count; range_index++) {
                svp_acl_data_buffer *buffer =
                    svp_acl_mdl_get_dataset_buffer(
                        models[model_index].inputs,
                        ranges[range_index].input_index);
                const unsigned char *address =
                    (const unsigned char *)svp_acl_get_data_buffer_addr(buffer);
                if (address == NULL) {
                    valid = 0;
                    snprintf(error, sizeof(error),
                             "snapshot range %zu input buffer is unavailable",
                             range_index);
                    break;
                }
                memcpy(copy + ranges[range_index].data_offset,
                       address + ranges[range_index].offset,
                       (size_t)ranges[range_index].length);
            }
        }
        if (!valid) {
            free(copy);
            return write_response_header(1, model_index, 0, error);
        }
        destroy_snapshot(&snapshots[snapshot_id - 1u]);
        snapshots[snapshot_id - 1u].valid = 1;
        snapshots[snapshot_id - 1u].model_index = model_index;
        snapshots[snapshot_id - 1u].range_count = range_count;
        snapshots[snapshot_id - 1u].byte_count = (size_t)byte_count;
        snapshots[snapshot_id - 1u].bytes = copy;
        memcpy(snapshots[snapshot_id - 1u].ranges, ranges,
               range_count * sizeof(ranges[0]));
        return write_response_header(0, model_index, 0, NULL);
    }
    if (opcode == OP_RESTORE_INPUTS) {
        uint32_t snapshot_id = input_count;
        resident_snapshot *snapshot;
        unsigned char *destinations[MAX_SNAPSHOT_RANGES] = {0};
        size_t range_index;
        if (output_count != 0 || write_count != 0 || snapshot_id == 0 ||
            snapshot_id > MAX_SNAPSHOTS) return -1;
        snapshot = &snapshots[snapshot_id - 1u];
        if (model_index >= model_count || !snapshot->valid ||
            snapshot->model_index != model_index) {
            snprintf(error, sizeof(error),
                     "snapshot %u is unavailable for model %u",
                     snapshot_id, model_index);
            return write_response_header(1, model_index, 0, error);
        }
        if (snapshot->bytes == NULL || snapshot->range_count == 0 ||
            snapshot->range_count > MAX_SNAPSHOT_RANGES) {
            snprintf(error, sizeof(error),
                     "snapshot %u has an invalid resident payload",
                     snapshot_id);
            return write_response_header(1, model_index, 0, error);
        }
        /* Resolve and validate the complete destination set before changing
         * any resident byte.  Snapshot metadata is executor-owned, but this
         * revalidation also protects against a malformed/stale in-memory slot. */
        for (range_index = 0; range_index < snapshot->range_count;
             range_index++) {
            resident_snapshot_range *range = &snapshot->ranges[range_index];
            svp_acl_data_buffer *buffer;
            if (range->input_index >= models[model_index].input_count ||
                range->length == 0 ||
                range->offset >
                    models[model_index].input_sizes[range->input_index] ||
                range->length >
                    models[model_index].input_sizes[range->input_index] -
                    range->offset ||
                range->data_offset > snapshot->byte_count ||
                range->length > snapshot->byte_count - range->data_offset) {
                snprintf(error, sizeof(error),
                         "snapshot %u restore range %zu is invalid",
                         snapshot_id, range_index);
                return write_response_header(1, model_index, 0, error);
            }
            buffer = svp_acl_mdl_get_dataset_buffer(
                models[model_index].inputs, range->input_index);
            destinations[range_index] =
                (unsigned char *)svp_acl_get_data_buffer_addr(buffer);
            if (destinations[range_index] == NULL) {
                snprintf(error, sizeof(error),
                         "snapshot %u restore buffer %zu is unavailable",
                         snapshot_id, range_index);
                return write_response_header(1, model_index, 0, error);
            }
        }
        for (range_index = 0; range_index < snapshot->range_count;
             range_index++) {
            resident_snapshot_range *range = &snapshot->ranges[range_index];
            memcpy(destinations[range_index] + range->offset,
                   snapshot->bytes + range->data_offset,
                   (size_t)range->length);
        }
        if (cached) {
            for (range_index = 0; range_index < snapshot->range_count;
                 range_index++) {
                resident_snapshot_range *range =
                    &snapshot->ranges[range_index];
                if (svp_acl_rt_mem_flush(
                        destinations[range_index] + range->offset,
                        (size_t)range->length) != SVP_ACL_SUCCESS) {
                    snprintf(error, sizeof(error),
                             "snapshot %u restore range %zu could not flush",
                             snapshot_id, range_index);
                    if (write_response_header(
                            2, model_index, 0, error) != 0) return -1;
                    return -2;
                }
            }
        }
        return write_response_header(0, model_index, 0, NULL);
    }
    if (opcode == OP_COPY_INPUTS) {
        resident_input_copy records[MAX_INPUT_COPY_RECORDS];
        size_t record_count = input_count;
        size_t record_index;
        if (model_index != 0 || output_count != 0 || write_count != 0 ||
            record_count == 0 || record_count > MAX_INPUT_COPY_RECORDS)
            return -1;
        memset(records, 0, sizeof(records));
        for (record_index = 0; record_index < record_count; record_index++) {
            unsigned char record[INPUT_COPY_RECORD_BYTES];
            uint32_t flags;
            int record_valid = 1;
            const char *record_error = NULL;
            persistent_model *destination_model = NULL;
            persistent_model *source_model = NULL;
            svp_acl_data_buffer *destination_buffer = NULL;
            svp_acl_data_buffer *source_buffer = NULL;
            unsigned char *destination_base = NULL;
            const unsigned char *source_base = NULL;
            if (read_exact_fd(STDIN_FILENO, record, sizeof(record)) != 0)
                return -1;
            records[record_index].destination_model = get_u32(record);
            records[record_index].destination_input = get_u32(record + 4);
            records[record_index].destination_offset = get_u64(record + 8);
            records[record_index].source_model = get_u32(record + 16);
            records[record_index].source_input = get_u32(record + 20);
            records[record_index].source_offset = get_u64(record + 24);
            records[record_index].length = get_u64(record + 32);
            flags = get_u32(record + 40);

            if (flags != 0 || records[record_index].length == 0 ||
                records[record_index].length > MAX_TENSOR_BYTES) {
                record_valid = 0;
                record_error = "has invalid flags or length";
            } else if (records[record_index].destination_model >= model_count ||
                       records[record_index].source_model >= model_count) {
                record_valid = 0;
                record_error = "names a model outside the current table";
            } else {
                destination_model =
                    &models[records[record_index].destination_model];
                source_model = &models[records[record_index].source_model];
                if (records[record_index].destination_input >=
                        destination_model->input_count ||
                    records[record_index].source_input >=
                        source_model->input_count) {
                    record_valid = 0;
                    record_error = "names an input outside its model";
                } else if (!input_copy_bounds_valid(
                               records[record_index].destination_offset,
                               destination_model->input_sizes[
                                   records[record_index].destination_input],
                               records[record_index].length) ||
                           !input_copy_bounds_valid(
                               records[record_index].source_offset,
                               source_model->input_sizes[
                                   records[record_index].source_input],
                               records[record_index].length)) {
                    record_valid = 0;
                    record_error = "exceeds source or destination input";
                }
            }
            if (record_valid) {
                destination_buffer = svp_acl_mdl_get_dataset_buffer(
                    destination_model->inputs,
                    records[record_index].destination_input);
                source_buffer = svp_acl_mdl_get_dataset_buffer(
                    source_model->inputs,
                    records[record_index].source_input);
                destination_base = (unsigned char *)
                    svp_acl_get_data_buffer_addr(destination_buffer);
                source_base = (const unsigned char *)
                    svp_acl_get_data_buffer_addr(source_buffer);
                if (destination_base == NULL || source_base == NULL) {
                    record_valid = 0;
                    record_error = "has an unavailable input buffer";
                }
            }
            if (record_valid) {
                records[record_index].destination = destination_base;
                records[record_index].source = source_base;
                records[record_index].same_buffer =
                    destination_base == source_base;
            } else if (valid) {
                valid = 0;
                snprintf(error, sizeof(error),
                         "input copy record %zu %s", record_index,
                         record_error == NULL ? "is invalid" : record_error);
            }
        }
        if (!valid)
            return write_response_header(1, 0, 0, error);

        /* Complete all cached-source synchronization before the first write.
         * A failure here therefore preserves every destination byte. */
        if (cached) {
            for (record_index = 0; record_index < record_count;
                 record_index++) {
                resident_input_copy *record = &records[record_index];
                if (svp_acl_rt_mem_invalidate(
                        (void *)(record->source + record->source_offset),
                        (size_t)record->length) != SVP_ACL_SUCCESS) {
                    snprintf(error, sizeof(error),
                             "input copy record %zu could not invalidate "
                             "source", record_index);
                    return write_response_header(2, 0, 0, error);
                }
            }
        }
        for (record_index = 0; record_index < record_count; record_index++) {
            resident_input_copy *record = &records[record_index];
            copy_input_bytes(
                record->destination + record->destination_offset,
                record->source + record->source_offset,
                (size_t)record->length, record->same_buffer);
        }
        if (cached) {
            for (record_index = 0; record_index < record_count;
                 record_index++) {
                resident_input_copy *record = &records[record_index];
                if (svp_acl_rt_mem_flush(
                        record->destination + record->destination_offset,
                        (size_t)record->length) != SVP_ACL_SUCCESS) {
                    snprintf(error, sizeof(error),
                             "input copy record %zu could not flush "
                             "destination", record_index);
                    return write_response_header(2, 0, 0, error);
                }
            }
        }
        return write_response_header(0, 0, 0, NULL);
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
        resident_scatter_f32_to_f16 records[MAX_MODEL_IO];
        size_t record_count = input_count;
        size_t record_index;
        /* An unbounded count is a framing failure and closes the stream.  All
         * bounded semantic failures below still drain their complete 48-byte
         * record list, return an error frame and leave the next request aligned. */
        if (record_count > MAX_MODEL_IO) return -1;
        memset(records, 0, sizeof(records));
        if (output_count != 0 || record_count == 0) {
            valid = 0;
            snprintf(error, sizeof(error),
                     "scatter frame requires records and no outputs");
        }
        for (record_index = 0; record_index < record_count; record_index++) {
            resident_scatter_f32_to_f16 *scatter = &records[record_index];
            unsigned char record[SCATTER_F32_TO_F16_RECORD_BYTES];
            uint32_t flags;
            int record_valid = 1;
            const char *record_error = NULL;
            persistent_model *destination_model = NULL;
            persistent_model *source_model = NULL;
            svp_acl_data_buffer *destination_buffer = NULL;
            svp_acl_data_buffer *source_buffer = NULL;
            if (read_exact_fd(STDIN_FILENO, record, sizeof(record)) != 0)
                return -1;
            scatter->destination_input = get_u32(record);
            scatter->source_model = get_u32(record + 4);
            scatter->source_output = get_u32(record + 8);
            flags = get_u32(record + 12);
            scatter->destination_base = get_u64(record + 16);
            scatter->destination_stride = get_u64(record + 24);
            scatter->channels = get_u32(record + 32);
            scatter->elements = get_u32(record + 36);
            if (flags != 0 || get_u64(record + 40) != 0 ||
                scatter->channels == 0 || scatter->elements == 0) {
                record_valid = 0;
                record_error = "has invalid flags or shape";
            } else if (model_index >= model_count ||
                       scatter->source_model >= model_count) {
                record_valid = 0;
                record_error = "names a model outside the current table";
            } else {
                destination_model = &models[model_index];
                source_model = &models[scatter->source_model];
                if (scatter->destination_input >=
                        destination_model->input_count ||
                    scatter->source_output >= source_model->output_count) {
                    record_valid = 0;
                    record_error = "names an input or output outside its model";
                } else if (!scatter_f32_to_f16_bounds_valid(
                               scatter->destination_base,
                               scatter->destination_stride,
                               scatter->channels, scatter->elements,
                               source_model->output_sizes[
                                   scatter->source_output],
                               destination_model->input_sizes[
                                   scatter->destination_input],
                               &scatter->source_bytes, &scatter->row_bytes,
                               &scatter->final_end)) {
                    record_valid = 0;
                    record_error = "exceeds source or destination";
                }
            }
            if (record_valid) {
                source_buffer = svp_acl_mdl_get_dataset_buffer(
                    source_model->outputs, scatter->source_output);
                destination_buffer = svp_acl_mdl_get_dataset_buffer(
                    destination_model->inputs, scatter->destination_input);
                if (source_buffer == NULL || destination_buffer == NULL) {
                    record_valid = 0;
                    record_error = "has an unavailable buffer";
                } else {
                    scatter->source = (const float *)
                        svp_acl_get_data_buffer_addr(source_buffer);
                    scatter->destination = (unsigned char *)
                        svp_acl_get_data_buffer_addr(destination_buffer);
                    if (scatter->source == NULL ||
                        scatter->destination == NULL) {
                        record_valid = 0;
                        record_error = "has an unavailable buffer";
                    } else {
                        scatter->source_capacity =
                            source_model->output_sizes[scatter->source_output];
                    }
                }
            }
            if (!record_valid && valid) {
                valid = 0;
                snprintf(error, sizeof(error), "scatter record %zu %s",
                         record_index,
                         record_error == NULL ? "is invalid" : record_error);
            }
        }
        if (!valid)
            return write_response_header(1, model_index, 0, error);

        /* Phase 2: every cached source becomes CPU-coherent before the first
         * destination byte is changed.  Failure preserves all destinations. */
        if (cached) {
            for (record_index = 0; record_index < record_count;
                 record_index++) {
                resident_scatter_f32_to_f16 *scatter =
                    &records[record_index];
                if (svp_acl_rt_mem_invalidate(
                        (void *)scatter->source, scatter->source_capacity) !=
                    SVP_ACL_SUCCESS) {
                    snprintf(error, sizeof(error),
                             "scatter record %zu could not invalidate source",
                             record_index);
                    return write_response_header(
                        2, model_index, 0, error);
                }
            }
        }

        /* Phase 3 has no fallible runtime calls: all records are converted only
         * after the entire request and cache-invalidate phase have succeeded. */
        for (record_index = 0; record_index < record_count; record_index++) {
            resident_scatter_f32_to_f16 *scatter = &records[record_index];
            scatter_f32_to_f16_rows(
                scatter->destination, scatter->source,
                scatter->destination_base, scatter->destination_stride,
                scatter->channels, scatter->elements);
        }

        /* Phase 4 publishes every modified row before ACK.  Once writes have
         * happened, a flush failure leaves coherence uncertain.  Send the
         * error frame, then terminate this executor session (return -2) so no
         * caller can accidentally continue without reloading all handles. */
        if (cached) {
            for (record_index = 0; record_index < record_count;
                 record_index++) {
                resident_scatter_f32_to_f16 *scatter =
                    &records[record_index];
                size_t channel;
                for (channel = 0; channel < scatter->channels; channel++) {
                    if (svp_acl_rt_mem_flush(
                            scatter->destination + scatter->destination_base +
                                channel * scatter->destination_stride,
                            (size_t)scatter->row_bytes) != SVP_ACL_SUCCESS) {
                        snprintf(
                            error, sizeof(error),
                            "scatter record %zu channel %zu could not flush "
                            "destination; executor session is poisoned",
                            record_index, channel);
                        if (write_response_header(
                                2, model_index, 0, error) != 0) return -1;
                        return -2;
                    }
                }
            }
        }
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
    resident_snapshot snapshots[MAX_SNAPSHOTS];
    size_t loaded_count = 0;
    svp_acl_error result;
    int acl_initialized = 0;
    int device_set = 0;
    int exit_code = 1;
    char ready_error[512] = {0};
    uintmax_t total_om_bytes = 0;
    memset(models, 0, sizeof(models));
    memset(snapshots, 0, sizeof(snapshots));
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
                                         options.cached, snapshots);
        if (serve_result == 1) {
            exit_code = 0;
            break;
        }
        if (serve_result != 0) break;
    }
cleanup:
    {
        size_t snapshot_index;
        for (snapshot_index = 0; snapshot_index < MAX_SNAPSHOTS;
             snapshot_index++) {
            destroy_snapshot(&snapshots[snapshot_index]);
        }
    }
    while (loaded_count != 0) destroy_model(&models[--loaded_count]);
    if (device_set) svp_acl_rt_reset_device(options.device_id);
    if (acl_initialized) svp_acl_finalize();
    close(protocol_output_fd);
    return exit_code;
}

#endif

# Persistent multi-model SVP_ACL executor for split PICO graphs (up to 128
# unique resident model handles; the MiniCPM5 startup/steady route table uses
# more than 64).
#
# Board build from this directory:
#   make \
#     SDK_ROOT=/path/to/SS928/sdk/smp/a55_linux/mpp/out \
#     CC=aarch64-mix210-linux-gcc

SDK_ROOT ?= /opt/ss928v100_gcc/smp/a55_linux/mpp/out
SVP_NPU_INC ?= $(SDK_ROOT)/include/svp_npu
SVP_NPU_LIB ?= $(SDK_ROOT)/lib/svp_npu
SVP_BASE_LIB ?= $(SDK_ROOT)/lib
CC ?= aarch64-mix210-linux-gcc
HOST_CC ?= cc
EXECUTOR_SRC ?= pico_persistent_acl_executor.c
EXECUTOR_BIN ?= ../bin/pico_persistent_acl_executor.aarch64
CONTRACT_CHECK_BIN ?= pico_persistent_acl_executor_contract_check
# These are the executor's direct runtime dependencies in the SS928 SDK.  The
# SDK ships protobuf-c as libprotobuf-c.so.1 without an unversioned .so link;
# naming the SONAME also avoids accidentally selecting the static archive.
SVP_ACL_LIBS ?= -lsvp_acl -lsvp_aicpu -l:libprotobuf-c.so.1 -lsecurec

CFLAGS += -std=c11 -O2 -Wall -Wextra -Werror -I$(SVP_NPU_INC)
LDFLAGS += -L$(SVP_NPU_LIB) -L$(SVP_BASE_LIB) \
	-Wl,-rpath-link,$(SVP_NPU_LIB) -Wl,-rpath-link,$(SVP_BASE_LIB)
LDLIBS += $(SVP_ACL_LIBS) -lpthread -ldl -lm

.PHONY: all clean contract-check syntax-check

all: $(EXECUTOR_BIN)

$(EXECUTOR_BIN): $(EXECUTOR_SRC)
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

contract-check: $(EXECUTOR_SRC)
	$(HOST_CC) -std=c11 -O2 -Wall -Wextra -Werror \
		-DPICO_PERSISTENT_ACL_CONTRACT_ONLY -o $(CONTRACT_CHECK_BIN) $<
	./$(CONTRACT_CHECK_BIN)

syntax-check: $(EXECUTOR_SRC)
	$(HOST_CC) -std=c11 -Wall -Wextra -Werror -fsyntax-only \
		-I$(SVP_NPU_INC) $<

clean:
	rm -f $(EXECUTOR_BIN) $(CONTRACT_CHECK_BIN)

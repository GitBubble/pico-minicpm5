# Reproducible executor build

[中文](EXECUTOR_BUILD.zh-CN.md)

`release/v0.2.0/release-manifest.json` binds the board executor by hash. This
document is the recipe that reproduces that hash from the source in this
repository, so the binding can be checked rather than trusted.

```text
source  app/native/pico_persistent_acl_executor.c
binary  sha256 cef4edb2ca71a3fd3b2f7ef9612d8090fb25fe95a19c465cd312383cf76a0374
        37,840 bytes, AArch64 ELF
```

## Toolchain

The sanctioned board family, run as an `linux/amd64` container:

```text
image     svp-pico-aarch64-tc:latest
compiler  aarch64-mix210-linux-gcc (HC&C V1R3C00SPC200B042_20221123) 7.3.0
prefix    /opt/linux/x86-arm/aarch64-mix210-linux/
flags     -O2
```

The SDK stub libraries are not redistributed here; supply them from your own
DDK installation at the path below.

## Command

```bash
docker run --rm --platform linux/amd64 -v "$PWD":/w svp-pico-aarch64-tc:latest sh -c '
  cd /w
  mkdir -p /tmp/lib
  cp <ddk>/acllib/lib64_aarch64-mix210-linux/stub/*.so* /tmp/lib/
  ln -sf libprotobuf-c.so.1 /tmp/lib/libprotobuf-c.so
  aarch64-mix210-linux-gcc -O2 \
    -I <ddk>/acllib/include/acl \
    app/native/pico_persistent_acl_executor.c \
    -L /tmp/lib -lsvp_acl -lsvp_aicpu -lprotobuf-c -lsecurec -lpthread -ldl -lm \
    -o pico_persistent_acl_executor.aarch64'
sha256sum pico_persistent_acl_executor.aarch64
```

## The one pitfall

Without the `libprotobuf-c.so -> libprotobuf-c.so.1` symlink the linker falls
back to the static `libprotobuf-c.a` in the stub directory. The build still
succeeds, the executor still runs, and the ELF is a different size with a
different hash. A correct build carries `NEEDED libprotobuf-c.so.1`.

## Why the binding changed in v0.2.0

`v0.1.0` pinned `b58b5c27…`, an executor that predates the resident-input
opcodes. The source in that release could not rebuild it, so the pin was a
statement about a binary nobody could re-derive. The v0.2.0 executor was built
three independent times, byte-identically, and this recipe was re-run against
the committed source before the manifest was written.

The new executor is what produced every number in
[the performance board](../release/perf/README.md): it retains the workspace
input across executes instead of rewriting it, which removes one full
workspace write per decode step. The saving is therefore proportional to the
context — `5.5 ms` at ctx1024, `27.6 ms` at ctx4096, `54.9 ms` at ctx8192,
all three landing at `3.7–4.7 GB/s` of avoided traffic.

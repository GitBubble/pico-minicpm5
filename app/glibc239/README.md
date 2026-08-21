# Sidecar glibc 2.39 (AArch64)

[中文](README.zh-CN.md)

Ubuntu 22.04 Jammy is glibc **2.35**. Community `libsvp_aicpu.so` records
`fmod@GLIBC_2.38`. Python and the rest of the image keep the system libc.

Only `bin/pico_persistent_acl_executor.community` runs under this loader:

```text
app/glibc239/ld-linux-aarch64.so.1 --library-path app/glibc239:... executor
```

Do not mix 2.35 and 2.39 in one process. The wrapper starts a second
loader; `chat.sh` stays on the system Python.

Objects are from Ubuntu 24.04 `libc6_2.39-0ubuntu8.8_arm64.deb`
(ports.ubuntu.com). Verify with `sha256sum -c SHA256SUMS`.

# Sidecar glibc 2.39（AArch64）

[English](README.md)

Ubuntu 22.04 Jammy 的 glibc 是 **2.35**。社区 `libsvp_aicpu.so` 记录了
`fmod@GLIBC_2.38`。Python 和系统其它进程继续用系统 libc。

只有 `bin/pico_persistent_acl_executor.community` 走这个 loader：

```text
app/glibc239/ld-linux-aarch64.so.1 --library-path app/glibc239:... executor
```

不要在同一个进程里混用 2.35 和 2.39。包装器另起一个 loader；`chat.sh`
仍用系统 Python。

目标文件来自 Ubuntu 24.04 `libc6_2.39-0ubuntu8.8_arm64.deb`
（ports.ubuntu.com）。校验：`sha256sum -c SHA256SUMS`。

# 板端 SVP ACL 运行库

[English](README.md)

这四只 AArch64 动态库是 `pico_persistent_acl_executor.aarch64` 真正链接的对
象。它们来自 `SS928V100_SDK_V2.0.2.2` 的 `smp/a55_linux/mpp/out/lib/`，随
`app/lib/` 一起交付，这样出厂 Euler Pi 镜像上解压部署包即可运行，不必再去翻
SDK。

| 文件 | 作用 |
|---|---|
| `libsvp_acl.so` | SVP ACL |
| `libsvp_aicpu.so` | AICPU 辅助 |
| `libprotobuf-c.so.1` | protobuf-c 1.x |
| `libsecurec.so` | 边界检查 C 库 |

厂方 `/opt/lib/npu` 是 Ascend / `libascendcl.so` 那一套，**不能**代替本目录。

`chat.sh` / `agent.sh` 在 Euler Pi 出厂镜像上把本目录放在
`LD_LIBRARY_PATH` 最前。Orange Pi AIfly / Pegasus 必须改用
`../lib-community/`（见该目录 README）。校验：`sha256sum -c SHA256SUMS`。

# Board SVP ACL runtime

[中文](README.zh-CN.md)

These four AArch64 shared objects are what
`pico_persistent_acl_executor.aarch64` actually links. They come from
`SS928V100_SDK_V2.0.2.2` (`smp/a55_linux/mpp/out/lib/`) and ship here so a
deployment tarball can run on a factory Euler Pi image without a second trip
through the SDK tree.

| File | Role |
|---|---|
| `libsvp_acl.so` | SVP ACL |
| `libsvp_aicpu.so` | AICPU helper |
| `libprotobuf-c.so.1` | protobuf-c 1.x |
| `libsecurec.so` | bounds-checked C |

Factory `/opt/lib/npu` is the Ascend/`libascendcl.so` stack and **cannot**
substitute for this directory.

`chat.sh` / `agent.sh` put `app/lib` first on `LD_LIBRARY_PATH` when
`libsvp_acl.so` is present. Verify with `sha256sum -c SHA256SUMS`.

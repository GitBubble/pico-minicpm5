# 社区版 SVP ACL 运行库（Pegasus / Orange Pi AIfly）

[English](README.md)

本目录是 Ubuntu 22.04 Jammy（glibc 2.35）+ Pegasus `ot_svp_npu` 的**社区**
用户态。不是 Euler Pi 商业套件 `../lib/`。

2026-08-21 chat 冒烟走过的路径是 **两个 glibc 进程共存**，不是在一个进程里
改绑 AICPU：

| 部件 | 作用 |
|---|---|
| `../glibc239/` | Ubuntu 24.04 `libc6` 2.39 sidecar；只给执行器进程 |
| `../bin/pico_persistent_acl_executor.community` | 包装器：`ld-linux-aarch64.so.1 --library-path glibc239:… community.bin` |
| `../bin/pico_persistent_acl_executor.community.bin` | 与 `.aarch64` 同一份 C 源，在板上链 Pegasus `libsvp_acl.a` + `libss_mpi.a` |
| `/usr/lib/svp_npu` | 板载 AICPU / ACL（要 `fmod@GLIBC_2.38`；sidecar 提供） |
| `libpico_mmz_anyaddr.so` | `LD_PRELOAD`：改写 `IOC_MMB_ALLOC_V3` 的 start，避免 OM 钉在已被占用的 MMZ 基址 |
| `libsvp_acl.so` / `libsvp_aicpu.so` / … | Pegasus 辅库；板载树不齐时的后备 |

`chat.sh` 在 Jammy、或存在 `/usr/lib/svp_npu` 时选这条路径；若存在
`/opt/ko/svp_npu` 则视为 Euler Pi 出厂镜像，改用商业 `app/lib`。

不要把社区板指到 `app/lib`：商业 ACL 在 12KB 的社区 `ot_svp_npu` 上会
`svp_acl_init ret=100000`。也不要对 `community.bin` 再 `LD_PRELOAD`
`pico_mpi_stubs`——桩会盖掉 ACL/MPI 符号，`svp_acl_init` 返回 `500004`。

`retarget_aicpu_glibc.py` 是走不了 sidecar 时的后备。它改写
`libm.so.6` 上那一条 VERNAUX（版本名 **和** ELF hash），让 Jammy 2.35
能加载 `libsvp_aicpu.so`。只改字符串、不改 `vna_hash` 时，`ld.so` 仍按
2.38 的 hash 匹配，会加载失败。已通过的冒烟用的是 sidecar + 板载 AICPU，
不是这条改绑。

```bash
python3 retarget_aicpu_glibc.py \
  --src /path/to/pegasus/.../out/lib/svp_npu/libsvp_aicpu.so \
  --dst ./libsvp_aicpu.so
cc -shared -fPIC -O2 -o libpico_mmz_anyaddr.so pico_mmz_anyaddr.c -ldl
sha256sum -c SHA256SUMS
```

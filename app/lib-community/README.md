# Community SVP ACL runtime (Pegasus / Orange Pi AIfly)

[中文](README.zh-CN.md)

This directory is the **community** userspace for Ubuntu 22.04 Jammy
(glibc 2.35) + Pegasus `ot_svp_npu`. It is **not** the Euler Pi commercial
set in `../lib/`.

The path that passed chat smoke (2026-08-21) is **two glibc processes**,
not a patched AICPU in one process:

| Piece | Role |
|---|---|
| `../glibc239/` | Ubuntu 24.04 `libc6` 2.39 sidecar; only the executor process |
| `../bin/pico_persistent_acl_executor.community` | wrapper: `ld-linux-aarch64.so.1 --library-path glibc239:… community.bin` |
| `../bin/pico_persistent_acl_executor.community.bin` | same C source as `.aarch64`, linked on the board against Pegasus `libsvp_acl.a` + `libss_mpi.a` |
| `/usr/lib/svp_npu` | board AICPU / ACL (needs `fmod@GLIBC_2.38`; the sidecar supplies it) |
| `libpico_mmz_anyaddr.so` | `LD_PRELOAD`: rewrite `IOC_MMB_ALLOC_V3` start so OM load is not pinned to a busy MMZ base |
| `libsvp_acl.so` / `libsvp_aicpu.so` / … | Pegasus extras; fallback if the board tree is incomplete |

`chat.sh` selects this path on Jammy / when `/usr/lib/svp_npu` is present,
unless `/opt/ko/svp_npu` marks an Euler Pi factory image.

Do **not** point a community board at `app/lib`. That commercial ACL
returns `svp_acl_init ret=100000` against the 12 KB community
`ot_svp_npu`. Do **not** `LD_PRELOAD` `pico_mpi_stubs` over
`community.bin` — those stubs override ACL/MPI symbols and
`svp_acl_init` returns `500004`.

`retarget_aicpu_glibc.py` is a fallback if you cannot run the sidecar.
It rewrites one `libm.so.6` VERNAUX (`GLIBC_2.38` name **and** ELF hash)
so Jammy 2.35 can load `libsvp_aicpu.so`. A name-only string patch leaves
`vna_hash` pointing at 2.38, and `ld.so` rejects the object. The smoke
that passed used the sidecar + board AICPU, not this retarget.

```bash
python3 retarget_aicpu_glibc.py \
  --src /path/to/pegasus/.../out/lib/svp_npu/libsvp_aicpu.so \
  --dst ./libsvp_aicpu.so
cc -shared -fPIC -O2 -o libpico_mmz_anyaddr.so pico_mmz_anyaddr.c -ldl
sha256sum -c SHA256SUMS
```

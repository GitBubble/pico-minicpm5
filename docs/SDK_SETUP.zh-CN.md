# 外部 SDK 配置

[English](SDK_SETUP.md)

本项目不分发 ATC、DDK、libinstsim、容器镜像或 `libsvp_custom.so`。
板端 *运行时* ACL 按两款产品分别交付：

| 板 | SDK | 本仓库运行时 |
|---|---|---|
| Euler Pi 2.0 | `SS928V100_SDK_V2.0.2.2` | `app/lib/` + `pico_persistent_acl_executor.aarch64` |
| Orange Pi AIfly | Pegasus / AIfly（`/usr/lib/svp_npu`，内核 `6.6.86-hi3403`） | `app/glibc239/` + `community.bin` + `app/lib-community/` |

见 `app/lib/README.zh-CN.md` 与 `app/lib-community/README.zh-CN.md`。不要混用
两套用户态。*编译* OM 用的 ATC/DDK 仍由用户自行提供。生产编译需要：支持
framework 5 / V101 / image-list / online OM / custom-op 的 ATC、匹配的
runtime/linker 库、注册 `ExtendRMSNorm` 的 custom-op 库，以及合法可用的
Hi3403 运行环境。

运行 `pico-minicpm5 doctor` 检查依赖。使用绝对 CLI 路径或厂商 SDK 初始化后的
环境。不得把 SDK、`.so`、板端凭据或 simulator dump 上传到公开 CI cache。

公开 CI 可运行 `pico-minicpm5 build --backend fake`，它只验证编排和 manifest
契约，不代表 PICO 数值正确或 OM 可上板加载。

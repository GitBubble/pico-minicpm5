# 外部 SDK 配置

[English](SDK_SETUP.md)

本项目不分发 ATC、DDK、libinstsim、板端动态库、容器镜像或
`libsvp_custom.so`。生产编译需要：支持 framework 5 / V101 / image-list / online
OM / custom-op 的 ATC、匹配的 runtime/linker 库、注册 `ExtendRMSNorm` 的
custom-op 库，以及合法可用的 Hi3403 运行环境。

运行 `pico-minicpm5 doctor` 检查依赖。使用绝对 CLI 路径或厂商 SDK 初始化后的
环境。不得把 SDK、`.so`、板端凭据或 simulator dump 上传到公开 CI cache。

公开 CI 可运行 `pico-minicpm5 build --backend fake`，它只验证编排和 manifest
契约，不代表 PICO 数值正确或 OM 可上板加载。

# 实验性 OM post-link

[English](README.md)

二进制 OM post-link 不属于受支持的默认流水线。已验收三句柄版本在 ONNX 图级
组合 24 层，再交给 ATC 编译整图。通用 linker 必须同时处理参数 heap 与累计表、
绝对分支、Item graph、TaskInfo/runtime table、workspace 和层间物理桥。

v0.1.0 不发布 post-linker。未来实现只有在通过 N=1 byte parity、N=2 liveness
与参数隔离、24 层公开输出 cosine `>0.98`、token exact 和板端验证后才能进入本目录。

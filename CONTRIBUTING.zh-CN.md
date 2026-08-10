# 贡献指南

[English](CONTRIBUTING.md)

提交必须可复现，并与私有工具链隔离。

1. 图或 manifest 改动必须增加 tiny fixture 单元测试。
2. 运行 `pytest` 和 `pico-minicpm5 release source --check-only`。
3. 不得提交权重、ONNX external data、OM、SDK、动态库、板端二进制、image list、
   原始板端 tensor 或凭据。
4. 用图、ABI 与数值合同描述编译器改动。默认合并路线是图级组合；实验性的
   二进制链接必须显式启用并 fail closed。

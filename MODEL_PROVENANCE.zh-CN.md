# 模型来源

[English](MODEL_PROVENANCE.md)

- 仓库：`openbmb/MiniCPM5-1B`
- revision：`4e9de7a0778dc1c362e983e6858f0e77542cbdca`
- 架构：标准 `LlamaForCausalLM`
- checkpoint shard SHA256：`7ab8fd86563125929be78aeec8cb3969c7ed2ead3be1ab9d3ec0a9fa69c8660d`
- shard 大小：`2,161,290,912` bytes；BF16 权重 payload：`2,161,265,664` bytes
- tokenizer SHA256：`3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`
- 派生 FP16 embedding SHA256：`5a93b589f0c5920021c95e04327c0771da2721d8eec2dd7ac1b283aa0d3b7df5`
- 固定 revision 的模型卡声明：Apache-2.0

项目通过 `hf download` 直接下载 checkpoint，不镜像模型文件。`model verify`
会拒绝 revision、geometry、symbol table、文件大小、config/index hash 或
safetensors header 不匹配的输入。ONNX、OM 和 embedding 均标记为模型派生产物。

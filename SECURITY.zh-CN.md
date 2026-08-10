# 安全说明

[English](SECURITY.md)

请先私下向维护者报告漏洞，再创建公开 issue。不得附带 checkpoint、SDK 归档、
板端凭据、私有动态库、mapper dump 或生产数据。

源码发布器拒绝 symlink、绝对路径、异常大文件和已知私有资产后缀。Hugging Face
token 仅由外部 `hf` 进程从环境读取，不通过命令行接收，也不写入 manifest。

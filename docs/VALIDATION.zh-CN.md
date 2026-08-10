# 验证阶梯

[English](VALIDATION.md)

1. **Checkpoint**：固定 revision、hash、geometry、symbol 与 BF16 span。
2. **图**：ONNX checker、唯一 namespace、无 dangling value，24 层为 5 输入/3 输出。
3. **深度递进**：N=1 identity；N=2 必须不同于 layer0 且匹配两层参考；再扩到
   4/8/12/24 层。
4. **本地执行**：公开 K/V/hidden 严格 cosine `>0.98`；每次运行后立即记录
   `runtime-capture.v1`，绑定 OM、build、position、ctx1024 和 raw hash。
5. **板端**：只加载三句柄；同 OM 的板端 raw 先对 libinstsim，再对 FP64。
6. **Head**：输入同 position 的 logical final hidden，logits cosine `>0.98` 且
   top-1 exact；residual 必须是 1536 FP32 全零。
7. **生成**：稳定边际 greedy exact、EOS、代码敏感 prompt 与多语言文本。
8. **性能**：记录 load time、每 token 延迟分布和 token/s。

`score` 会结构化识别 K/V 顺序，必要时解码 C4 hidden，并在门禁失败时返回非零。
聚合 packed cosine 是 Release 门槛，同时必须输出逐层 slice 诊断，防止低能量层被
聚合结果掩盖。

两种 scorer 都强制接收 `--om`、`--build-manifest` 和 `--capture-manifest`。
qualification 校验 head 输入 hash 等于同 position transformer 的 logical hidden，
并拒绝跨 build、跨 context、假 compiler、未绑定 raw 文件或可降低的阈值。

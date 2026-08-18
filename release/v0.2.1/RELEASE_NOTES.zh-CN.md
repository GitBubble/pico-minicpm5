# pico-minicpm5 v0.2.1

本次源码发行加入在 8192、10240 与 16384 token 上运行 owner-built MiniCPM5
decode 产物所需的长上下文合同；不重新分发模型权重、授权 SDK 动态库或本地编译的
OM 文件。

## 主要变化

- `ctx8192` 已资格化：冻结的 48-token greedy oracle 全量一致，全部公开输出
  cosine 严格高于 `0.98`；修正后的官方 EOS 序列以句号后接 EOS 结束。
- `ctx10240`、`ctx16384` profile 已接入，但仍 fail-closed 保持 `pending`；受控
  测试仍需显式启用未资格化 profile。
- 三个 profile 的 4097-token head-skip 用例均通过（8192/10240/16384 摄入耗时
  602.48/681.89/910.42 秒）。ctx10240 因冻结 greedy 用例仅 36/48、尾部 hidden
  cosine `0.978842` 而保持 pending；ctx16384 最佳重标定尾部 hidden/K/V 仅为
  `0.957146/0.985295/0.967172`，同样保持 pending。
- 8192/10240/16384 decode 合同会自动保留各 context 对应的 workspace 输入，
  并在加载前严格校验 7 输入描述符。
- teacher-forced prompt 摄入会跳过除最后一个已知 prompt 位置之外的词表 head。
  ctx8192 的 4097-token 门跳过 4096 次 head 调用并精确输出预期下一 token。
- `app/agent.sh` 支持 `CONTEXT_PROFILE=ctx8192|ctx10240|ctx16384`；环境变量与
  命令行选择冲突时 fail-closed。
- 加入可选 eager tool-output prefill；默认关闭，并受 resident K/V 与 prefix
  reuse 前置条件保护。

Release workflow 发布 Python sdist/wheel、SPDX SBOM 与校验和；板端 OM 仍由其
owner 单独提供并完成资格化。

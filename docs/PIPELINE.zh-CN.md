# 端到端流水线

[English](PIPELINE.md)

## 1. 下载并冻结源模型

```bash
pico-minicpm5 model fetch --local-dir work/model
pico-minicpm5 model verify --model-dir work/model
```

下载固定 revision `4e9de7a0778dc1c362e983e6858f0e77542cbdca`。验证阶段检查
config、index、safetensors 大小/header/hash、模型 geometry 和符号契约；不匹配
时 fail closed。认证只读取环境变量 `HF_TOKEN`。

## 2. 构建浮点参考

```bash
pico-minicpm5 reference capture \
  --model-dir work/model --out work/reference --context 1024
```

参考数据来自官方 checkpoint，记录各 position、各层 hidden/K/V 与 logits。发布
门禁使用 FP64 greedy oracle；reference manifest 必须绑定 checkpoint hash、context、
prompt token 和生成工具版本。

## 3. 重新生成 ATC calibration

```bash
pico-minicpm5 reference calibrate \
  --reference work/reference --family decode --out work/calibration/decode
pico-minicpm5 reference calibrate \
  --reference work/reference --family prefill --out work/calibration/prefill
```

decode 与 position-0 prefill 的量化域不同，必须独立标定。样本打包顺序必须与
24 层图的 K/V axis 和层顺序一致。calibration manifest 记录输入 hash、clip 契约
与 context；不得用 decode donor 替代 prefill 标定。

## 4. 导出真实权重 ONNX

```bash
pico-minicpm5 onnx export-layers \
  --model-dir work/model --out work/onnx/layers --context 1024
pico-minicpm5 onnx export-head \
  --model-dir work/model --out work/onnx/head/model.onnx
```

导出的 24 个 decoder layer 使用真实 safetensors 权重，覆盖 ExtendRMSNorm、Q/K/V/O
投影、RoPE、KV append、GQA attention、SwiGLU MLP 与 residual。head 为最终
RMSNorm 加词表投影。随机权重 fixture 只能用于前端测试，不能生成 Release。

族特定的 Clip 节点钉住激活量程契约；这些边界如何以 `min(推断值, Clip 边界)`
封顶 ATC 的 IFMR 量程搜索、以及两个 family 为何不能合并，见
[量化契约](QUANTIZATION_CONTRACT.zh-CN.md)。

## 5. 组合 24 层

```bash
pico-minicpm5 onnx compose \
  --layers-dir work/onnx/layers/decode --family decode \
  --out work/onnx/decode/model.onnx \
  --pack-input-kv --pack-output-kv --external-data
pico-minicpm5 onnx compose \
  --layers-dir work/onnx/layers/prefill --family prefill \
  --out work/onnx/prefill/model.onnx \
  --pack-input-kv --pack-output-kv --external-data
```

composer 为每层增加 namespace，以 SSA 串联 hidden，共享 mask/RoPE，并用 Slice /
Concat 将 ABI 固定为 5 个公共输入和 3 个输出。大 initializer 每个使用独立 external
file、offset 0，避免 protobuf 2 GiB 与 ATC signed-offset 限制。

## 6. 编译

```bash
pico-minicpm5 build \
  --decode-onnx work/onnx/decode/model.onnx \
  --prefill-onnx work/onnx/prefill/model.onnx \
  --head-onnx work/onnx/head/model.onnx \
  --calibration work/calibration --out work/om --context 1024 \
  --atc /path/to/atc --custom-ops-lib /path/to/libsvp_custom.so
```

ATC/DDK/custom-op 必须由用户合法安装。build manifest 记录 backend、context、输入
ONNX/calibration/toolchain hash、三只 OM 的大小与 SHA。fake backend 只验证 CI
编排，不能用于数值资格。

## 7. 执行、评分、资格与发布

每只 OM 执行完成后立即创建 runtime capture，将实际 OM、build manifest、position
和 raw file hash 绑定。prefill position 0、decode position 1 的示例：

```bash
pico-minicpm5 capture --runner libinstsim --role prefill \
  --position 0 --context 1024 --om work/om/prefill.om \
  --build-manifest work/om/build-manifest.json \
  --output work/prefill/out.0.bin --output work/prefill/out.1.bin \
  --output work/prefill/out.2.bin --report work/prefill/runtime-capture.json
pico-minicpm5 score --position 0 --context 1024 \
  --output work/prefill/out.0.bin --output work/prefill/out.1.bin \
  --output work/prefill/out.2.bin --reference work/reference \
  --om work/om/prefill.om --build-manifest work/om/build-manifest.json \
  --capture-manifest work/prefill/runtime-capture.json \
  --report work/prefill-score.json
```

decode 使用相同形式并将 role/position 改为 `decode/1`。head 必须使用相同 position
transformer 产生的 logical FP32 `next_hidden`，residual 必须是 1536 个 FP32 全零：

```bash
pico-minicpm5 capture --runner libinstsim --role head_flat \
  --position 1 --context 1024 --om work/om/head_flat.om \
  --build-manifest work/om/build-manifest.json \
  --input hidden=work/head/next_hidden.f32.bin \
  --input residual=work/head/residual_zero.f32.bin \
  --output work/head/logits.f32.bin --report work/head/runtime-capture.json
pico-minicpm5 score-head --position 1 --context 1024 \
  --output work/head/logits.f32.bin --reference work/reference \
  --om work/om/head_flat.om --build-manifest work/om/build-manifest.json \
  --capture-manifest work/head/runtime-capture.json \
  --hidden-input work/head/next_hidden.f32.bin \
  --residual-input work/head/residual_zero.f32.bin \
  --report work/head-score.json
```

最终绑定三份分数并组包：

```bash
pico-minicpm5 qualify --prefill-score work/prefill-score.json \
  --decode-score work/decode-score.json --head-score work/head-score.json \
  --models work/om --out work/qualification.json
pico-minicpm5 release assemble --models work/om --model-dir work/model \
  --qualification work/qualification.json --out artifacts/release
pico-minicpm5 release verify artifacts/release
```

固定门槛为公开 tensor cosine 严格 `>0.98`、head logits `>0.98` 且 top-1 exact，
再执行 greedy token、EOS、多语言和板端性能门禁。runtime capture 是本地可审计
hash 血缘，不是密码学远程证明；`--runner` 的真实性由操作者负责。

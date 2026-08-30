# 一颗 NPU 上的两个模型：视觉 skill

[English](MULTIMODAL_VISION.md)

状态：板端已验证。实现为 `app/src/vision_jobs.py`、`app/src/vision_worker.py`
和 `app/src/minicpm4v_vision.py`；运行期入口是 `describe_image` 与
`--vision-queue`。

## 1. 为什么是队列而不是函数调用

MiniCPM5-1B 负责回答，MiniCPM-4v-0.5B 负责看图。两者各自常驻三个 OM 句柄，
而 NPU 只有一颗。如果 agent 直接内联调用视觉流水线，它就会在看图期间彻底停止
应答——在这块板上看一张图要 `21.5 s`，而打个招呼只要 `3.2 s`。

所以两者之间连的是作业队列，不是函数调用。`describe_image` 写下一条作业、
约一毫秒内返回作业号；独立的 `vision_worker` 进程认领并去看；答案在之后的
某个回合浮现。语言模型从不为看图阻塞。

队列就是一个装 JSON 文件的目录，没有别的——没有要维持的守护进程，没有要绑定
的套接字。作业以原子 rename 走 `queued → claimed → done|failed`，所以 worker
崩了留下的是一条**看得见、可重排**的 claimed 作业，而不是丢失的作业；两个进程
可以各自独立重启。

## 2. 硬件拒绝了什么，以及怎么绕过去

发布了四个句柄，能装三个。`decode.om` 声明 53 个输入、49 个输出——5 个，加上
每层 K 和 V 各一个端口——而本 SDK 对单模型的端口上限是 32，因此它在装载期就被
拒绝。这块板上根本没有可用的 KV 缓存解码步。

这看起来对生成是致命的，其实不是。`prefill_decode.om` 会输出整个 200 行窗口的
logits，所以下一个词可以从最后一个真实位置读出来——再下一个词，就是把前一个
拼上去重跑一次 prefill。生成变成了**反复 prefill**：

```text
vision.om ──▶ resample.om ──▶ 64 个视觉 token
                                   │
              ┌────────────────────┘
              ▼
     prefill_decode.om  ◀── PRE(9) + 图像(64) + MID(3) + 问题 + POST(6) + 已生成部分
              │
              └──▶ logits[prefill_len - 1] ──▶ argmax ──▶ 追加，重复
```

窗口 200 行、问题很短，所以大约还能放下一百个词的答案。填满窗口的调用者会
**收到报错而不是被静默截断**——悄悄丢掉最新那个词会让循环永远停不下来。

代价是每个词一次完整的 200 行 prefill，即 `0.52 s`，平直，且降不下去。
这正是队列存在的理由。

## 3. 实测

提交到完成，1440×900 截图，40 词上限，Hi3403：

| 阶段 | |
|---|---|
| 被 worker 领取 | `1.02 s` |
| 首词可见 | `1.98 s` |
| 节奏 | `0.52 s`/词，平直 |
| 完成 | `22.56 s` |

`1.02 s` 的领取延迟是 worker 的轮询间隔（`--poll-seconds`），不是模型开销。
预处理、`vision.om` 和 `resample.om` 加起来不到一秒，且每张图只付一次，
不是每个词付一次。

原样输出：

```text
这张图片展示了一个名为"HISpark"的软件界面。在顶部，可以看到一个名为"Model"的选项，
并且有一个"+"号按钮，这可能用于添加或创建…（达到 40 词上限）
```

## 4. 答案边写边到

"在回合边界送达"不等于流式，而第一版确实不是流式。两个缺口，都已补上：

`report_vision` 只在 `input()` 返回之后才跑，所以用户问完图之后如果不再打字，
就什么都看不到——REPL 阻塞在一个不会被"生成完成"打断的读上。现在提示符改为
轮询：`select` 等 stdin、超时就原地重绘一行进度。非 tty 时没有东西可动画，
所以管道、测试和录制走的仍是普通读取。

而 worker 原来只在最后写一次盘——`21.5 s` 静默之后吐出一整段。现在
`VisionQueue.progress` 每生成一个词就发布一次。记录保持 `claimed` 状态，
所以它是**在看**而不是**送达**：`collect()` 仍然忽略它，`finish()` 会清掉它，
不会重复显示两次。

还有两个窗口不刷新，且都是有意为之：语言模型在流式输出自己的回复时不重绘视觉
行；用户敲下第一个字符后，轮询让位给 `input()`。

## 5. 声明工具不是免费的

工具 schema 是固定的提示词前缀，凡是提到它的回合都要按 prefill token 计费。
因此 `describe_image` 单独占一个只含一个工具的 profile，而不是并入读取集；
并且只有**同时**满足两个条件才声明：这一回合有视觉意图**且**点名了文件。
`看看这个目录` 和不带文件名的 `看看这张图` 都留在读取集上。

它也只在**存在 worker** 的部署上才声明。没有 `--vision-queue` 时，这个工具
被调用的唯一结局是被拒绝，而 schema 照样计费，所以这类回合退回读取集——
后者至少还能说清那个文件是什么。

## 6. 怎么跑

worker 只持有 4v 的句柄，别的什么都不管：

```sh
# $QUEUE 是两个进程都能写的任意目录；$VLM 放三个 4v 句柄外加 tokenizer.json
# 和 token_emb.bin；$EXE 是按 docs/EXECUTOR_BUILD.md 构建的常驻 ACL 执行器。
python3 -u src/vision_worker.py \
  --queue "$QUEUE" \
  --model-dir "$VLM" \
  --executable "$EXE" \
  --library-path /opt/lib/svp_npu --library-path /opt/lib --library-path /opt/lib/npu \
  --poll-seconds 1.0 --max-new 40
```

agent 指向同一个目录：

```sh
./agent.sh --vision-queue "$QUEUE"
```

`--max-new` 是时延预算，不是质量旋钮：每个词都是一次完整 prefill，
所以 40 词是 21 秒，80 词就是 42 秒。

## 7. 预处理契约

`minicpm4v_vision.py` 是照着已公布的 C++ 契约移植的，不是猜的：512×512 CUBIC、
`(x/255 − 0.5)/0.5`、把 16×16 patch reshape 写成五轴转置、200 token 模板及其
两遍注意力掩码，以及贪心最长匹配词表——该表**不是** BPE。`MASK_MIN_VALUE` 是
`-9999999.0`，从头文件里读出来的，而不是想当然的 `-10000`。300 MB 的嵌入表用
`seek` 按行读，从不整体加载。

有两个协议细节值得重复，因为在钉死之前它们都造成过静默的数据损坏：执行器
**先写完所有输出尺寸，再写所有 payload**——交错读会把第二个尺寸从第一个张量的
字节里解析出来；以及请求头里的 `public_inputs` 是**模型的公共输入端口数**，
不是写入的个数。

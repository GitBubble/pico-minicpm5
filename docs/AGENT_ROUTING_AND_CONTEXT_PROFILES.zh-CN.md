# Agent 路由与运行时 Context Profile 设计

[English](AGENT_ROUTING_AND_CONTEXT_PROFILES.md)

状态：设计已确认；Iteration 1 以本文为实现基线。

## 1. 目标

SS928 应用不能把每个请求都无条件交给语言模型。命令处理、权限、工具执行、
结果展示和 MiniCPM5 推理必须解耦；同一份应用源码通过严格校验的 runtime
profile 支持多个静态编译 context。

设计以板端实际成本为依据：列目录约 3 ms；当前 ctx1024 三句柄版本的一个
transformer/head 位置约 106.6 ms；重放系统提示、工具定义和历史会主导首 token
时间；K/V 存储和 attention 工作量随编译 context 增长。

MiniCPM5 只负责语言理解、规划、推理和总结。命令、参数范围、权限、路径边界及
已经完备的结构化结果展示由确定性代码负责。

## 2. 路由流水线

每个请求生成一个可审计的决策：

```text
LOCAL_COMMAND       本地 REPL 命令，不发送给模型
DIRECT_TOOL         确定性工具调用，工具结果就是最终答案
TOOL_THEN_MODEL     工具提供事实，MiniCPM 负责总结或推理
MODEL_ONLY          不需要工具定义和工具执行
PLAN_AND_APPROVE    模型提出步骤，宿主逐项校验和授权
CLARIFY             缺少的选择会实质改变操作，必须澄清
```

返回策略独立于工具：

```text
DIRECT_RAW
DIRECT_FORMATTED
MODEL_SUMMARIZE
MODEL_REASON
CONFIRM_BEFORE_EXECUTE
```

例如：

```json
{
  "mode": "DIRECT_TOOL",
  "confidence": 0.99,
  "tool_calls": [
    {
      "name": "list_directory",
      "arguments": {"path": ".", "max_entries": 10}
    }
  ],
  "response_policy": "DIRECT_FORMATTED",
  "schema_profile": "none",
  "permission": "automatic",
  "reason": "explicit directory listing request"
}
```

### 2.1 路由层级

1. `/help`、`/context`、`/max`、`/clear`、`/quit` 等本地命令不调用工具和模型。
2. 明确的列出、显示、读取、搜索、status 请求调用只读工具并直接展示结果。
3. 包含总结、解释、比较、诊断、建议的请求先取事实，再将紧凑证据交给模型。
4. 含糊或多步骤任务使用 MiniCPM5 原生 XML 工具协议规划，宿主校验每次调用。
5. 写入、shell 及未来网络工具继续遵守显式权限；路由不能扩大授权。

## 3. 工具注册与渐进披露

每个工具声明副作用、权限、结果类型、输出预算、超时和是否允许直接展示。
工具按组注册：

```text
filesystem-read  list_directory, read_file, search_text
git-read         git_status 及未来 diff/log
filesystem-write write_file
shell            run_shell
```

模型只看到相关工具组。`MODEL_ONLY` 不注入工具 schema，直接路由也完全不渲染
schema。工具结果由类型元数据、紧凑预览和稳定 result_id 组成；大结果分页读取，
不整段复制进上下文。

## 4. Runtime Profile 合同

Context 不是一个孤立整数。Runtime profile 必须同时绑定：

- 编译 context 与 past length；
- decode、position-zero/prefill、head 产物；
- packed K/V 几何和 runtime descriptor 数量；
- 默认和最大生成长度；
- 压缩阈值和预留预算；
- chat/agent/tool 能力；
- 工具轮数与输出预算；
- 产物资格状态和数值阈值。

Profile 在进程启动时选择：

```bash
./app/chat.sh --profile ctx128
./app/agent.sh --profile ctx1024
./app/agent.sh --profile ctx4096
./app/agent.sh --profile ctx8192
```

也可直接指定文件：

```bash
./app/agent.sh --profile /opt/pico-minicpm5/profiles/ctx4096.json
```

命令行 `--profile` 优先于可选的 `PICO_PROFILE`，后者优先于安装默认值；初始
安装默认保持 ctx1024。

### 4.1 最终能力矩阵

| Profile | Chat | Agent | 定位 |
|---|---:|---:|---|
| ctx128 | 支持 | **不支持** | 短对话/补全、低内存、冒烟 |
| ctx1024 | 支持 | 支持 | 默认本地 Agent |
| ctx4096 | 支持 | 支持 | 文档、代码、多步任务 |
| ctx8192 | 支持 | 支持 | 超长上下文和长会话 |

执行 `agent.sh --profile ctx128` 必须在加载模型句柄前失败，并提示改用
`chat.sh --profile ctx128`。

### 4.2 静态 ABI 与内存

24 层、每层 2 个 KV head、head_dim=128、FP16 cache 时，单个 packed K 或 V：

```text
48 * (context - 1) * 128 * 2 bytes
```

| Context | 单个 K/V | K+V |
|---:|---:|---:|
| 128 | 1,560,576 B（约 1.49 MiB） | 约 2.98 MiB |
| 1024 | 12,570,624 B（约 12 MiB） | 约 24 MiB |
| 4096 | 50,319,360 B（约 48 MiB） | 约 96 MiB |
| 8192 | 100,651,008 B（约 96 MiB） | 约 192 MiB |

加载器必须验证 profile context、attention mask 宽度、K/V past length 与 runtime
`--context` 完全一致；发布 profile 还必须校验产物 hash。禁止静默截断、根据文件名
猜合同或跨 profile 复用不匹配的 OM。ABI/hash 相同时 tokenizer、embedding 和
vocabulary head 可以共享。

### 4.3 生成长度

Context capacity 与回答策略分离。Profile 提供 `default_max_new`、
`max_new_limit` 和 `reserve_tokens`。`/max` 同时显示配置范围、硬件范围和本轮可用
预算；ctx8192 不意味着默认回答 8191 tokens。

## 5. SS928 Context 执行策略

只增加 context 而不改变 prompt 执行会让 Agent 更慢。按约 106.6 ms/position，
重放 4096 或 8192 个位置约需 7.3 或 14.6 分钟。因此 Agent profile 必须具备：

1. 会话级 resident K/V 追加，而不是每轮从 position 0 重放；
2. 固定 system/tool 前缀 K/V 快照；
3. 明确工具结果直接返回，不调用模型；
4. 紧凑、带类型、可分页的工具证据；
5. 接近 profile 阈值时执行 context rebase；
6. 分开报告 route、tool、prompt/prefill、decode 和总耗时。

长期 native compiler 增加 S16/S32/S128 等真正多 token prefill。Builder 由 context
和 sequence length 参数化，不为每个 context 复制应用源码。

## 6. Context Rebase

会话接近阈值时保留固定前缀、小型 task-state 摘要、最近对话和工具产物引用，
删除旧的原始工具输出，然后只重建一次紧凑 transcript。`/clear` 仍是显式全量重置。

第一阶段不支持会话内热切 context。切换需要加载 handle、重新分配 K/V、迁移
cache layout、校验 mask/RoPE 并重新做数值门禁。未来热迁移必须显式执行，并与
全量回放参考做 token-exact 对照。

## 7. 权限与终端安全

直接路由不改变权限。只读工具可自动执行；`write_file` 和 `run_shell` 仍对精确
参数逐次授权。路径限制在 workspace，拒绝 symlink 逃逸，不可信工具结果在终端
展示前转义控制字符。

## 8. 可观测性与性能目标

每个请求记录：

```text
route_mode, route_reason, route_ms, tool_ms, model_called,
prompt_tokens_new, prompt_tokens_replayed, prefix_cache_hit,
prefill_ms, decode_ms, generated_tokens, time_to_first_token_ms, total_ms
```

初始常驻目标：

| 请求 | 目标 |
|---|---:|
| 本地命令 | <10 ms |
| 直接目录/git status | <50 ms |
| 直接文件/搜索窗口 | <200 ms |
| 工具后简短模型总结 | 前缀/KV 优化后 3～10 s |

终端对直接路由明确显示 `model skipped`，并把 prompt 处理时间和输出速度分开。

## 9. 迭代计划

1. **Iteration 1：** 发布本文；实现 profile 加载与 ctx128 Chat-only 门禁；目录结果
   直接返回；增加 route/tool 耗时证据。
2. **Iteration 2：** 渐进披露工具组，增加 typed/paged result 引用。
3. **Iteration 3：** 保留会话 live K/V，增加固定前缀快照和 token-exact 回放门禁。
4. **Iteration 4：** 实现 context rebase 和长会话门禁。
5. **Iteration 5：** 为 ctx4096/8192 编译并验证真正多 token prefill。

每个发布 context 都必须通过 descriptor 校验、public-output cosine 严格大于 0.98、
greedy token、边界位置、EOS、context overflow、板端加载和性能报告。

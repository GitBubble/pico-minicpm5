# 上下文账本

[English](CONTEXT_LEDGER.md)

状态：已接受的设计；实现是 `app/src/context_ledger.py`。

## 1. 为什么要有它

在已验收的 ctx1024 板端，送入一个 prompt token 要 `79.49 ms`；而整段对话——系统
提示、工具 schema、历史、工具结果、当前这一轮——必须一起装进 1024 个 token 里。
两条都是硬约束，可在此之前哪一条都无法在运行时观测：agent 会报告这次请求执行了
多少个 prompt token，却从不报告这些 token 花在了什么上面。

第一次把这个问题直接问出来，答案并不是谁预想的那个。复现一次板端实测轮次的
形状——历史里已经有一次目录列举，用户说了句「你好」：

| 段 | token | 占比 | 送入耗时 |
|---|---:|---:|---:|
| 工具 schema，8 个工具 | 481 | 61.8% | 38.23 s |
| 系统提示正文 | 193 | 24.8% | 15.35 s |
| 历史里的工具结果 | 80 | 10.3% | 6.36 s |
| 对话本身 | 24 | 3.1% | 1.92 s |
| `<s>` | 1 | 0.1% | 0.08 s |

三分之二的上下文、三十八秒的延迟，都花在向一个只是被要求打个招呼的模型描述工具
上。这正是账本要让人看见的缺陷，而且换任何别的方式都看不见：上面每一个数都是
对的，可它们一个都不在任何日志里。

## 2. 精确，而非估算

通过 API 与模型对话的 agent 框架只能估算 token 数，因为 tokenizer 在网络那一头。
我们的 tokenizer 就在进程里，而且运行时**本来就要**把拼好的 prompt 编码一次去喂
模型。所以归属可以是精确的，而且可以不额外花钱。

机制是偏移归属。tokenizer 为每个 token 返回一对 `(start, end)` 字符偏移；每个
token 归给持有它**首字符**的那一段。分段各自编码既更慢又是错的——tokenizer 会跨
段边界做合并，各部分之和不等于整体。

跨越边界的 token 只计一次，计在它起始的那一段，并把这类 token 的个数作为
`boundary_tokens` 报出来。它是整套核算里唯一的歧义来源，所以被测量而不是被藏起来。

## 3. 分段

段是从已渲染的 prompt 里、沿 MiniCPM5 线格式**读回来**的，而不是由拼装它的代码
交下来。这让账本与渲染器解耦——包括用另一种语言写的渲染器——也让段表本身可校验：
段必须精确铺满整个字符串，凡是没有任何段认领的字符，`measure` 都会报出来。

```text
preamble           开头的 <s>
system             系统消息，扣除工具块
tool_schema        <tools> … </tools> 块，按其中的工具命名
history_user       更早的一轮用户输入
history_assistant  更早的一轮助手输出
tool_result        一个 <tool_response> 块
current_user       正在问的这一轮
generation_prompt  结尾的助手前缀
```

## 4. 成本与压力

常驻 K/V 前缀在 token 位置上永远是一个前缀，所以「已在缓存里的 token」与「必须
送入的 token」这条切分是精确的，不需要按比例摊。给定该 profile 实测的
`prompt_token_ms`，每一段都会报出它那些**新增** token 要花的秒数。整段常驻的段
成本为零——这正是固定前缀快照值钱的原因，也正是前缀抖动昂贵的原因。

压力是 `total_tokens / (capacity - reserve_tokens)`。它越过 1.0 的时刻早于模型
拒绝任何东西的时刻，所以它是该做 rebase 的信号。

## 5. 契约

记录带 `schema: pico.minicpm5.context-ledger.v1`。每条记录都成立三条定律，每条都
有对应的测试：

1. **守恒**：各段 token 数之和等于 `total_tokens`。没有任何段认领的部分就是其中
   一行，kind 为 `unattributed`；它的数值同时另记在标量 `unattributed_tokens`
   里，所以核对总数的人不会把它加两遍。
2. **常驻切分**：各段 `new_tokens` 之和等于 `total_tokens - resident_tokens`。
3. **铺满**：段之间永不重叠，也不越过文本末尾。违反其一的段表抛
   `LedgerError`，而不是产出一个看起来合理的错数。

## 6. 可移植性

本设计借鉴的那个框架，是通过服务图来计量 token 的。我们刻意不这么做。
`context_ledger.py` 除 `dataclasses` 与 `re` 之外什么都不 import：不 import
tokenizer，不 import agent，不 import 运行时。它的输入是一个字符串、一张段表，以及
每个 token 一对偏移；输出是一个普通字典。

扫描是对偏移的单趟遍历配一个指向有序段表的游标——`O(n_tokens + n_segments)`，每个
token 零分配。预期路径是用 Rust 实现同一趟扫描，并按本项目一贯的方式设门：两个
实现对同一输入各出一份 `pico.minicpm5.context-ledger.v1` 记录，两份记录必须一致。
schema 就是契约，测试就是一致性套件。

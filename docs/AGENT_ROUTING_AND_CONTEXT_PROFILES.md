# Agent routing and runtime-context profiles

[中文](AGENT_ROUTING_AND_CONTEXT_PROFILES.zh-CN.md)

Status: accepted design; Iteration 1 is the implementation baseline.

## 1. Goals

The SS928 application must not treat every user request as an unconditional
language-model generation. Host commands, permissions, tool execution, result
presentation and MiniCPM5 reasoning are separate stages. The same application
source supports multiple statically compiled context variants through a
validated runtime profile.

The design optimizes for the actual SS928 cost model:

- a directory listing takes about 3 ms;
- a transformer/head position takes about 106.6 ms in the accepted ctx1024
  three-handle build;
- replaying a large system/tool/history prompt dominates time to first token;
- K/V storage and attention work grow with the compiled context.

The model is therefore reserved for language understanding, planning,
reasoning and summarization. Deterministic code owns commands, parameter
bounds, permissions, path confinement and presentation of already-complete
structured results.

## 2. Routing pipeline

Each request produces one auditable decision:

```text
LOCAL_COMMAND       local REPL command; never sent to the model
DIRECT_TOOL         deterministic tool call; result is the final answer
TOOL_THEN_MODEL     tool supplies facts; MiniCPM summarizes or reasons
MODEL_ONLY          no tool schema or tool execution is needed
PLAN_AND_APPROVE    model proposes steps; each side effect is validated/approved
CLARIFY             a missing choice would materially change the operation
```

The response policy is independent of the tool:

```text
DIRECT_RAW
DIRECT_FORMATTED
MODEL_SUMMARIZE
MODEL_REASON
CONFIRM_BEFORE_EXECUTE
```

Example decision:

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

### 2.1 Routing levels

1. Local commands such as `/help`, `/context`, `/max`, `/clear` and `/quit`
   complete without a model or tool.
2. Explicit display/list/read/search/status requests use deterministic
   read-only tools and return their structured result directly.
3. Requests containing summarize, explain, compare, diagnose or recommend run
   the necessary tool and send only its compact evidence to MiniCPM5.
4. Ambiguous multi-step tasks use MiniCPM5's native XML tool-call contract as a
   planner. The host validates every call.
5. Mutation, shell and future network tools retain explicit permission policy;
   routing never grants authority.

## 3. Tool registry and progressive disclosure

Every tool records its side effect, permission, result type, output budget,
timeout and whether direct presentation is legal. Tool definitions are grouped:

```text
filesystem-read  list_directory, read_file, search_text
git-read         git_status and future diff/log tools
filesystem-write write_file
shell            run_shell
```

Only the relevant group is rendered into the model prompt. `MODEL_ONLY`
requests receive no tool schema. Direct routes execute without rendering any
schema. Tool results use typed metadata, a compact preview and a stable result
identifier; large evidence is paged instead of copied into the conversation.

## 4. Runtime-profile contract

A context is not a standalone integer. A runtime profile binds:

- compiled context and past length;
- decode, position-zero/prefill and head artifacts;
- packed K/V geometry and runtime descriptor counts;
- default and maximum generation limits;
- compaction threshold and reserved budget;
- chat/agent/tool capabilities;
- tool-round and output budgets;
- artifact qualification state and numeric threshold.

Profiles are selected at process startup:

```bash
./app/chat.sh --profile ctx128
./app/agent.sh --profile ctx1024
./app/agent.sh --profile ctx4096
./app/agent.sh --profile ctx8192
```

An explicit path is also valid:

```bash
./app/agent.sh --profile /opt/pico-minicpm5/profiles/ctx4096.json
```

Command line selection has precedence over an optional `PICO_PROFILE`, which
has precedence over the installed default. The initial installed default
remains ctx1024.

### 4.1 Final capability matrix

| Profile | Chat | Agent | Intended workload |
|---|---:|---:|---|
| ctx128 | yes | **no** | short chat/completion, low memory, smoke tests |
| ctx1024 | yes | yes | default local agent |
| ctx4096 | yes | yes | documents, code and multi-step tasks |
| ctx8192 | yes | yes | long-context tasks and long sessions |

Attempting `agent.sh --profile ctx128` fails before model handles are loaded and
directs the operator to `chat.sh --profile ctx128`.

### 4.2 Static ABI and memory

For 24 layers, two K/V heads per layer, head dimension 128 and FP16 cache, one
packed K or V input occupies:

```text
48 * (context - 1) * 128 * 2 bytes
```

| Context | One K or V | K + V |
|---:|---:|---:|
| 128 | 1,560,576 B (~1.49 MiB) | ~2.98 MiB |
| 1024 | 12,570,624 B (~12 MiB) | ~24 MiB |
| 4096 | 50,319,360 B (~48 MiB) | ~96 MiB |
| 8192 | 100,651,008 B (~96 MiB) | ~192 MiB |

The loader must fail closed unless profile context, attention-mask width, K/V
past length and runtime `--context` agree. It must also validate artifact hashes
once a profile is published. No silent truncation, inferred filename contract or
cross-profile OM reuse is allowed. Tokenizer, embedding and vocabulary head may
be shared when their ABI and hashes are identical.

### 4.3 Generation limits

Context capacity and response policy are distinct. A profile provides
`default_max_new`, `max_new_limit` and `reserve_tokens`. `/max` reports the
configured range, the hardware context range and the currently available
budget. Larger contexts do not imply an 8191-token default answer.

## 5. Context processing on SS928

Increasing context without changing prompt execution makes the agent slower.
At roughly 106.6 ms per replayed position, replaying 4096 or 8192 positions
would take approximately 7.3 or 14.6 minutes. Agent profiles therefore require:

1. append-only per-session resident K/V rather than replaying from position 0;
2. fixed system/tool-prefix K/V snapshots;
3. direct tool routes that do not invoke the model;
4. compact typed tool evidence and pagination;
5. context rebase near a profile-defined threshold;
6. separate reporting for route, tool, prompt/prefill, decode and total time.

Longer-term native compiler work adds true multi-token prefill families such as
S16, S32 and S128. The builder is parameterized by context and sequence length;
the source application is not forked per context.

## 6. Context rebase

When an agent session approaches its threshold, it retains the fixed prefix, a
small task-state summary, recent turns and references to tool artifacts. Raw
old tool output is not retained. The compact transcript is then rebuilt once.
`/clear` remains an explicit full reset.

Hot switching context is out of scope for the first implementation. Profiles
are selected at startup because changing context requires model-handle loading,
K/V reallocation, cache layout migration, mask/RoPE validation and a numerical
gate. A future migration must be explicit and token-exact against a replay
reference.

## 7. Permissions and terminal safety

Direct routing never changes the permission model. Read-only operations may run
automatically. `write_file` and `run_shell` require approval bound to the exact
arguments. Paths remain confined to the configured workspace, symlink escapes
are rejected and untrusted result bytes are escaped before terminal display.

## 8. Observability and performance objectives

Every request report records:

```text
route_mode, route_reason, route_ms, tool_ms, model_called,
prompt_tokens_new, prompt_tokens_replayed, prefix_cache_hit,
prefill_ms, decode_ms, generated_tokens, time_to_first_token_ms, total_ms
```

Initial resident targets:

| Request | Target |
|---|---:|
| local command | <10 ms |
| direct directory/git status | <50 ms |
| direct file/search window | <200 ms |
| tool plus short model summary | 3-10 s after prefix/KV work |

The terminal explicitly prints `model skipped` for direct routes and separates
prompt processing from generation throughput.

## 9. Iteration plan

1. **Iteration 1:** publish this contract, add profile loading and ctx128
   chat-only enforcement, implement direct directory results, and add route/tool
   timing evidence.
2. **Iteration 2:** progressively disclose tool groups and add typed/paged
   result references.
3. **Iteration 3:** retain live session K/V and add fixed-prefix snapshots with
   token-exact replay parity tests.
4. **Iteration 4:** add context rebase and long-session gates.
5. **Iteration 5:** compile and qualify true multi-token prefill families for
   ctx4096 and ctx8192.

Each published context must pass descriptor validation, public-output cosine
strictly greater than 0.98, greedy-token gates, boundary positions, EOS, context
overflow behavior, board load and performance reporting.

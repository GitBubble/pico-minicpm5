# Run the prebuilt demo directly on Hi3403

[中文说明](README.zh-CN.md)

This directory is the board-user entry point. It assumes the files from the
GitHub `v0.1.0` release have already been copied to
`/opt/pico-minicpm5`; no ONNX export, ATC compilation or host-side
Python package is needed.

## Expected board layout

```text
/opt/pico-minicpm5/
├── app/
│   ├── chat.sh
│   ├── agent.sh
│   ├── bin/pico_persistent_acl_executor.aarch64
│   ├── native/{Makefile,pico_persistent_acl_executor.c}
│   ├── profiles/{ctx128,ctx1024,ctx4096,ctx8192}.json
│   └── src/{merged_board_server.py,minicpm_agent.py,
│            minicpm_profile.py,
│            pico_minicpm5_split_board_runner.py,probe_om_execute_latency.py,
│            qualify_minicpm_greedy_chain.py}
├── models/{prefill.om,decode.om,head_flat.om}          # qualified ctx1024
├── models/ctx128/{prefill.om,decode.om}                # when qualified
├── models/ctx4096/decode.om                            # qualified; prefill is the shared models/prefill.om
├── models/ctx8192/decode.om                            # pending (strict-EOS gate); shared prefill
└── assets/{token_embedding.f16.bin,tokenizer.json}
```

The licensed Hi3403 runtime libraries are expected in
`/root/pico_default_smoke/lib`. They are supplied by the board SDK and are not
part of this repository.

## Run on the board

```bash
cd /opt/pico-minicpm5
chmod +x app/chat.sh app/agent.sh app/bin/pico_persistent_acl_executor.aarch64

# Official MiniCPM5 chat template, without tools.
./app/chat.sh

# Native tool-calling agent. The three handles load only once.
./app/agent.sh

# Explicit profile selection. ctx128 is chat-only.
./app/chat.sh --profile ctx128
./app/agent.sh --profile ctx4096
```

```text
        /\_/\
       ( o.o )    HiAgent
        > ^ <     Hi3403 端侧 AI
     本地运行 · 隐私安全 · 实时响应

⠹ Loading three resident model handles  6.4s
✓ ready · loaded 3 handles · ctx1024 · 7.4s
Agent ready · /help · /tools · /think on|off · /context · /clear · /quit
You ❯ Read the first 20 lines of README.md and summarize the project.
⠴ Planning  0.8s
⚙ read_file(path='README.md', start_line='1', end_line='20')
✓ read_file: 1: # pico-minicpm5
MiniCPM ✦ ...
You ❯ /quit
```

Both board applications default to the qualified `ctx1024` profile. `chat.sh` uses the official
MiniCPM5 chat template without tool definitions and retains conversation
history until `/clear`. `agent.sh` adds the native tool protocol described
below.
Tool definitions are rendered inside `<tools>`, the model emits its trained
`<function>/<param>` XML, and results return through `<tool_response>`. The
agent retains conversation/tool history until `/clear`; model handles,
executor and device buffers remain resident.

Built-ins are `list_directory`, `read_file`, `search_text`, `git_status`,
`write_file` and `run_shell`. The first four run automatically. Writes and
shell commands prompt `Allow once? [y/N]` every time and default to deny. File
tools are confined to the startup working directory; use `--workspace PATH`
to set an explicit boundary. `/tools`, `/permissions` and `/context` display
the registry, policy and token budget.

`/help` prints Linux-style command help with syntax, ranges and scope. Use
`/help COMMAND` (for example, `/help max`) for the detailed form.

| Command | Purpose and scope |
|---|---|
| `/help [COMMAND]` | List every local command or explain one command. |
| `/profile` | Show the runtime profile, context and capability; switching requires restart. |
| `/tools` | Show registered native tools without executing them. |
| `/permissions` | Show which tools are automatic and which require approval. |
| `/think [on\|off]` | Query or change thinking for subsequent Agent generations. |
| `/context` | Show the selected profile's prompt usage, including tools and history. |
| `/clear` | Clear conversation/tool history without reloading model handles; `/reset` is an alias. |
| `/max [N]` | Query or set the profile response limit, further limited by remaining context. |
| `/quit` | Close the resident session; `/exit` and Ctrl-D are equivalent. |

The agent knows the configured workspace root and uses `path='.'` for it, so it
must inspect available paths rather than ask the user for the current directory.
Unambiguous requests for the current directory, a directory listing, a literal
file window, literal text search or Git status are routed directly to read-only
tools and displayed with `model skipped`; general selection, summaries and
transformations remain model-native. The qualified ctx1024 profile caps tool
responses at 800 characters;
directory, file and search defaults are deliberately small and can be narrowed
or paged with a follow-up call.

Model-native requests use progressive tool disclosure. Clearly read-only work
receives only read schemas; explicit write or command intent adds the matching
permission-gated tool, while ambiguous development tasks retain all tools as a
fail-safe. Successful results carry a type, local reference, truncation state
and next offset. `read_result_page` retrieves another bounded page without
re-running the original operation; the most recent 16 results are retained.

Chat and Agent enable token-exact resident session-K/V reuse by default, so a
later turn executes only newly appended prompt tokens. `/clear` resets both the
conversation and resident-prefix metadata. Use `REUSE_SESSION_KV=0
./app/chat.sh` or `REUSE_SESSION_KV=0 ./app/agent.sh` for a full-replay diagnostic
baseline. In a two-turn board Agent A/B, reuse was token-exact with replay and a
134-token prefix hit reduced turn-two latency from `94.94 s` to `80.78 s`.
Contextual follow-ups such as “what does the second line do?” omit unrelated
tool schemas when the evidence is already in the transcript; mutation and shell
intents remain fail-closed behind their permission schemas.

The current source also provides lazy fixed system/tool-prefix snapshots per
schema. A new generic resident-input snapshot/restore executor opcode stores
only the K/V rows actually used; schema switches and `/clear` can restore them
without returning cache bytes to Python. This path is enabled by default for
Agent after a board token-exact A/B: restoring a 137-token prefix took
`1.76 ms`, reduced the repeated 32-token request from `26.97 s` to `12.56 s`
(`53.4%`), and preserved the exact token IDs and text. Use
`FIXED_PREFIX_SNAPSHOTS=0 ./app/agent.sh` only for a full-replay diagnostic.

At a profile's `compact_at_tokens`, Agent performs deterministic context
rebase. Raw historical tool output is replaced by its typed local reference,
the current exchange and recent turns remain byte-exact, and
`reserve_tokens` preserves response headroom. Reports record token counts
before/after and the number of compacted turns. If the current exchange alone
still cannot fit, the runtime fails closed and asks for a shorter request or
`/clear`. A Hi3403 long-session board A/B compacted 12 old tool turns from
`2808` to `810` prompt tokens in both runs and produced the exact same
`[18655, 4569, EOS]` response. On the repeated run, a 643-token resident-prefix
hit reduced total time from `69.45 s` to `14.61 s` (`4.75x`).

Known prompt tokens do not execute the vocabulary head: until the final prompt
position, the runtime runs only transformer and K/V update, because each next
input token is already known. The head and argmax still run on the final prompt
position and every generated token. This preserved the exact board output
while reducing the same cold long-prompt request from `86.70 s` to `69.45 s`
(`19.89%`) and the resident-prefix repeat from `18.17 s` to `14.61 s`
(`19.59%`). Per-position reports mark this with `head_skipped`.

Request reports also include a fail-closed `prefill_schedule`. Its canonical
policy is `S128 -> S32 -> S16 -> strict S1 tail`, but the current qualified
bundle enables only S1. Wider families are never selected merely because an
OM file exists; each context-specific artifact must first pass descriptor,
public-output cosine `>0.98`, K/V publication, prefill-to-decode handoff,
token-exact and Hi3403 board gates.

An operator can verify an optional release-v4 activation at application
startup with `--prefill-activation-manifest` plus the live
`--available-bytes`, `--base-resident-bytes` and `--reserve-bytes` values.
All four options are required together. `/profile` and JSON reports expose
both qualification state and executable widths. This release has no wide
production handler registered: the typed dispatcher is fake-transport tested,
but no complete wide OM has passed release gates and there is no CLI injection
path. Qualified S16/S32/S128 artifacts therefore remain unavailable to the
scheduler and execution stays strict S1; no wide label is simulated.
Release-v4 token-exact evidence binds the actual head OM and embedding, and
startup rehashes those files with both S1 route OMs, the imported protocol
runner, executor, descriptors and registered wide OMs immediately before
spawn. The deployment tree must remain trusted and read-only/immutable for the
process lifetime; this path-based preflight does not claim an inherited-fd
handoff against an active writer. See
[native-prefill release qualification](../docs/NATIVE_PREFILL_RELEASE_QUALIFICATION.md).

Agent thinking is disabled by default. Start it enabled with either
`./app/agent.sh --thinking` or `THINKING=1 ./app/agent.sh`. During a resident
session, `/think` reports the state and `/think on` or `/think off` changes the
next generation without reloading the three models. Thinking tokens consume
the same ctx1024 budget as tool definitions, history and the final answer.

A timed activity indicator covers model loading and time-to-first-token, then
the final answer streams token by token. The ctx1024 default response limit is
128 tokens; `/max N` displays or changes the active profile limit without
restarting the models. For ctx1024, `N` may be 1–1023;
the effective output also depends on tool definitions, prompt, history and
tool results. Older completed turns are cleared when needed; a turn gets at
most four tool rounds by default.

Colour and animation are enabled only on an interactive terminal. During the
initial model load, HiAgent scans and blinks while MiniCPM uses a gentler blink;
the animation reuses loading time and adds no startup delay. Later planning and
generation retain the single-line activity indicator, so old transcript text is
never overwritten. Redirected, piped and log output remains stable plain text. Use
`NO_COLOR=1 ./app/agent.sh` to disable colour, or pass `--no-spinner` to disable
animation. `--color always|never|auto` and `PICO_MINICPM5_COLOR` explicitly
select the colour policy. The REPL hides low-level executor loading logs by
default; pass `--verbose-executor` when debugging them.

The executor starts loading all three OMs before the tokenizer is parsed,
overlapping two independent cold-start costs. Qualified runtime profiles also
bind transformer output slots as K=0, V=1 and hidden=2, eliminating the former
four-execute, roughly 0.8-second KV startup probe. The ctx1024 Hi3403 generation
smoke now loads in `7.4 s`, down from `8.2–8.6 s` after overlap alone and
`10.8–11.5 s` originally. Legacy model arguments without a trusted slot contract
retain dynamic probing as a compatibility fallback.

For a single non-interactive prompt:

```bash
./app/chat.sh --prompt 'The capital of France is' --max-new 16

# Chinese
./app/chat.sh --prompt '请用一句话解释什么是神经网络。' --max-new 32

# Arithmetic / EOS path
./app/chat.sh --prompt '1+1 equals' --max-new 16
```

These `--prompt` examples retain the raw-completion path. With no arguments,
`chat.sh` starts the official no-tools chat REPL and `agent.sh` starts the
native agent. Explicit `chat.sh --interactive` retains the legacy raw prompt
REPL.

Both REPLs use an append-only UTF-8-safe decoder. CJK characters split across
token boundaries are buffered until complete, preventing output stalls and
whole-answer replay. GNU readline handles UTF-8 line editing; coloured prompt
escapes are marked zero-width so backspace can erase the complete input line.

Both launchers accept extra server arguments after the script name and
recognize these environment overrides:

| Variable | Default | Purpose |
|---|---|---|
| `PICO_MINICPM5_ROOT` | parent of `app/` | Deployment root |
| `PICO_RUNTIME_LIB` | auto-detect | Board runtime libraries |
| `PYTHON` | auto-detect | Python executable |
| `TOKENIZERS` | empty | Optional extra `site-packages` path |
| `PROMPT` | unset | Optional one-shot prompt; unset starts REPL |
| `PICO_PROFILE` | `ctx1024` | Runtime profile selected before model loading |
| `MAX_NEW` | profile default | Optional initial maximum generated tokens |
| `THINKING` | `0` | `agent.sh` startup thinking: `0/1`, `off/on`, `false/true` |
| `PICO_MINICPM5_COLOR` | `auto` | `auto`, `always` or `never` |
| `NO_COLOR` | unset | Disable ANSI colour while in auto mode |

Runtime libraries are detected first at `/root/pico_default_smoke/lib`, then
at `/opt/ss928-runtime/lib`. Python is detected first at
`$PICO_MINICPM5_ROOT/venv/bin/python`, then as `python3`.

## Quick checks

```bash
cd /opt/pico-minicpm5
sha256sum -c SHA256SUMS
test -r "${PICO_RUNTIME_LIB:-/opt/ss928-runtime/lib}/libsvp_acl.so" || \
  ls "${PICO_RUNTIME_LIB:-/opt/ss928-runtime/lib}"
python3 -c 'import tokenizers; print(tokenizers.__version__)'
```

If `tokenizers` is installed elsewhere, set `TOKENIZERS` to its
`site-packages` directory. If a runtime library cannot be loaded, set
`PICO_RUNTIME_LIB` to the directory containing the matching board SDK
libraries. The supplied executor is AArch64; its source and Makefile are under
`app/native/` for toolchain-specific rebuilding:

```bash
cd /opt/pico-minicpm5/app/native
make SDK_ROOT=/path/to/sdk/smp/a55_linux/mpp/out CC=aarch64-mix210-linux-gcc
```

The runtime-profile and hybrid-routing contract is documented in the source
repository's [Agent routing and runtime-context profile design](https://github.com/GitBubble/pico-minicpm5/blob/main/docs/AGENT_ROUTING_AND_CONTEXT_PROFILES.md).
ctx128 is deliberately chat-only. ctx4096 and ctx8192 remain pending until
their exact OM sets pass descriptor, numeric (`>0.98`) and board gates;
controlled development requires the explicit `--allow-unqualified-profile`.

The optimized ctx1024 release measured `105.5–106.1 ms/token`, or
`9.42–9.48 token/s`, with 48/48 greedy tokens exact.

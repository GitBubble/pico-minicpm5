# Run the prebuilt demo directly on SS928

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
│   └── src/{merged_board_server.py,minicpm_agent.py,
│            pico_minicpm5_split_board_runner.py,probe_om_execute_latency.py,
│            qualify_minicpm_greedy_chain.py}
├── models/{prefill.om,decode.om,head_flat.om}
└── assets/{token_embedding.f16.bin,tokenizer.json}
```

The licensed SS928 runtime libraries are expected in
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
```

```text
        /\_/\
       ( o.o )    MiniCPM 5
        > ^ <     SS928 local AI
     ctx1024 · resident KV · streaming

⠹ Loading three resident model handles  6.4s
✓ ready · loaded 3 handles · ctx1024 · 10.2s
Agent ready · /help · /tools · /think on|off · /context · /clear · /quit
You ❯ Read the first 20 lines of README.md and summarize the project.
⠴ Planning  0.8s
⚙ read_file(path='README.md', start_line='1', end_line='20')
✓ read_file: 1: # pico-minicpm5
MiniCPM ✦ ...
You ❯ /quit
```

Both board applications default to `ctx1024`. `chat.sh` uses the official
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

The agent knows the configured workspace root and uses `path='.'` for it, so it
must inspect available paths rather than ask the user for the current directory.
Unambiguous directory-listing requests are routed directly to the read-only
`list_directory` tool; general tool selection remains model-native.
Because this release is ctx1024, tool responses are capped at 800 characters;
directory, file and search defaults are deliberately small and can be narrowed
or paged with a follow-up call.

Agent thinking is disabled by default. Start it enabled with either
`./app/agent.sh --thinking` or `THINKING=1 ./app/agent.sh`. During a resident
session, `/think` reports the state and `/think on` or `/think off` changes the
next generation without reloading the three models. Thinking tokens consume
the same ctx1024 budget as tool definitions, history and the final answer.

A timed activity indicator covers model loading and time-to-first-token, then
the final answer streams token by token. The default
response limit is 128 tokens; `/max N` displays or
changes it without restarting the models. For ctx1024, `N` may be 1–1023;
the effective output also depends on tool definitions, prompt, history and
tool results. Older completed turns are cleared when needed; a turn gets at
most four tool rounds by default.

Colour and animation are enabled only on an interactive terminal. Redirected,
piped and log output automatically remains stable plain text. Use
`NO_COLOR=1 ./app/agent.sh` to disable colour, or pass `--no-spinner` to disable
animation. `--color always|never|auto` and `PICO_MINICPM5_COLOR` explicitly
select the colour policy. The REPL hides low-level executor loading logs by
default; pass `--verbose-executor` when debugging them.

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
| `MAX_NEW` | `128` | Initial maximum generated tokens |
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

The optimized ctx1024 release measured `105.5–106.1 ms/token`, or
`9.42–9.48 token/s`, with 48/48 greedy tokens exact.

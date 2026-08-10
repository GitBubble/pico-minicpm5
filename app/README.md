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
│   ├── bin/pico_persistent_acl_executor.aarch64
│   ├── native/{Makefile,pico_persistent_acl_executor.c}
│   └── src/{merged_board_server.py,pico_minicpm5_split_board_runner.py,
│            probe_om_execute_latency.py,qualify_minicpm_greedy_chain.py}
├── models/{prefill.om,decode.om,head_flat.om}
└── assets/{token_embedding.f16.bin,tokenizer.json}
```

The licensed SS928 runtime libraries are expected in
`/root/pico_default_smoke/lib`. They are supplied by the board SDK and are not
part of this repository.

## Run on the board

```bash
cd /opt/pico-minicpm5
chmod +x app/chat.sh app/bin/pico_persistent_acl_executor.aarch64

# Start the resident REPL. The three model handles load only once.
./app/chat.sh
```

```text
MiniCPM5 REPL ready. Commands: /help, /reset, /quit
You> The capital of France is
MiniCPM>  Paris, ...
You> /quit
```

Each REPL prompt starts a fresh logical ctx1024 sequence; the model handles,
executor process and allocated device buffers stay resident. This avoids the
roughly 10-second model reload between questions. `/reset` marks a new
transcript in an optional JSON report. The current REPL is independent-turn
completion, not a chat-template conversation-history implementation.

For a single non-interactive prompt:

```bash
./app/chat.sh --prompt 'The capital of France is' --max-new 16

# Chinese
./app/chat.sh --prompt '请用一句话解释什么是神经网络。' --max-new 32

# Arithmetic / EOS path
./app/chat.sh --prompt '1+1 equals' --max-new 16
```

`chat.sh` accepts extra server arguments after the script name and recognizes
these environment overrides:

| Variable | Default | Purpose |
|---|---|---|
| `PICO_MINICPM5_ROOT` | parent of `app/` | Deployment root |
| `PICO_RUNTIME_LIB` | auto-detect | Board runtime libraries |
| `PYTHON` | auto-detect | Python executable |
| `TOKENIZERS` | empty | Optional extra `site-packages` path |
| `PROMPT` | unset | Optional one-shot prompt; unset starts REPL |
| `MAX_NEW` | `24` | Maximum generated tokens |

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

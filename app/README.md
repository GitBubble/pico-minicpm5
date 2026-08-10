# Run the prebuilt demo directly on SS928

[中文说明](README.zh-CN.md)

This directory is the board-user entry point. It assumes the files from the
GitHub `v0.1.0` release have already been copied to
`/root/minicpm5_gate_3handle`; no ONNX export, ATC compilation or host-side
Python package is needed.

## Expected board layout

```text
/root/minicpm5_gate_3handle/
├── app/chat.sh
├── models/{prefill.om,decode.om,head_flat.om}
├── assets/{token_embedding.f16.bin,tokenizer.json}
├── bin/pico_persistent_acl_executor.resident.aarch64
└── src/{merged_board_server.py,pico_minicpm5_split_board_runner.py,
         probe_om_execute_latency.py,qualify_minicpm_greedy_chain.py}
```

The licensed SS928 runtime libraries are expected in
`/root/pico_default_smoke/lib`. They are supplied by the board SDK and are not
part of this repository.

## Run on the board

```bash
cd /root/minicpm5_gate_3handle
chmod +x app/chat.sh bin/pico_persistent_acl_executor.resident.aarch64

# English
PROMPT='The capital of France is' MAX_NEW=16 sh app/chat.sh

# Chinese
PROMPT='请用一句话解释什么是神经网络。' MAX_NEW=32 sh app/chat.sh

# Arithmetic / EOS path
PROMPT='1+1 equals' MAX_NEW=16 sh app/chat.sh
```

`chat.sh` accepts extra server arguments after the script name and recognizes
these environment overrides:

| Variable | Default | Purpose |
|---|---|---|
| `GATE` | `/root/minicpm5_gate_3handle` | Deployment root |
| `LIB` | `/root/pico_default_smoke/lib` | Board runtime libraries |
| `TOKENIZERS` | `/opt/pico-minicpm5/venv/lib/python3.10/site-packages` | Python package path |
| `PROMPT` | `The capital of France is` | Input prompt |
| `MAX_NEW` | `24` | Maximum generated tokens |

## Quick checks

```bash
cd /root/minicpm5_gate_3handle
sha256sum -c SHA256SUMS
test -r "${LIB:-/root/pico_default_smoke/lib}/libsvp_acl.so" || \
  ls "${LIB:-/root/pico_default_smoke/lib}"
python3 -c 'import tokenizers; print(tokenizers.__version__)'
```

If `tokenizers` is installed elsewhere, set `TOKENIZERS` to its
`site-packages` directory. If a runtime library cannot be loaded, set `LIB` to
the directory containing the matching board SDK libraries. The supplied
executor is AArch64; its source and Makefile are under `native/` in the runtime
archive for toolchain-specific rebuilding.

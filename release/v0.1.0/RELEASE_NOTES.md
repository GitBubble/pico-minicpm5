# pico-minicpm5 v0.1.0

[中文](RELEASE_NOTES.zh-CN.md)

This source release captures the reproducible MiniCPM5-1B → ONNX → packed
24-layer PICO OM workflow and the accepted SS928 ctx1024 three-handle artifact
contract.

The corresponding qualified artifacts are identified by size and SHA256 in
`release-manifest.json`. They are not embedded in the public source archive.
The portable `qualification.json` retains raw-output hashes, public tensor
cosines, greedy token evidence and performance while omitting the board address.
`release assemble` can build a local model bundle from user-supplied artifacts
only after every hash and policy check passes.

The default compiler route is graph-level 24-layer composition followed by one
ATC invocation per family. Binary OM post-linking is not a production path.

## Runtime refresh — 2026-08-10

The three qualified OM files remain byte-identical. The board application now
keeps packed K/V resident, scatters current FP32 rows directly to FP16 cache,
packs C4 embeddings directly and prepares RoPE sparsely. SS928 throughput rises
from the original `8.20–8.60 token/s` range to `9.42–9.48 token/s` while keeping
48/48 greedy tokens exact and preserving EOS and Chinese prompt results.

Board application source, executor C and its Makefile now live under `app/`.
The compiled executor is carried only by the runtime archive; duplicate
standalone executor source/binary/Makefile assets are retired.

The runtime now also exposes a resident stdin REPL. Running `app/chat.sh`
without arguments loads the three handles once and accepts repeated prompts;
`/help`, `/reset` and `/quit` are built in. One-shot `--prompt` execution stays
compatible. REPL responses now stream as tokens are generated, default to a
128-token limit and support `/max N`; reaching the response or context limit
is reported explicitly instead of silently truncating text.

Tool calling is exposed as a separate `app/agent.sh` application. It
uses the official `<tools>/<function>/<param>/<tool_response>` chat contract,
supports multi-step tool feedback and conversation history, confines file
tools to a workspace, and asks before every write or shell command. `chat.sh`
remains the plain conversational REPL, and one-shot `--prompt` stays available.
The chat entry now uses the official no-tools chat template. UTF-8-safe
incremental decoding buffers incomplete CJK byte pieces, fixing the terminal
stall and whole-answer replay previously seen around split characters.
Agent thinking remains opt-in and is now configurable at startup with
`--thinking`/`THINKING=1` or per resident session with `/think on|off`.
Readline-aware coloured prompts now erase UTF-8 input cleanly back to column
zero instead of leaving the first character visible.
The agent prompt now binds `.` to the configured workspace and requires direct
filesystem inspection instead of asking the user for a current path. Compact
800-character tool results preserve enough ctx1024 budget for a final answer.
Unambiguous listing requests have a read-only deterministic route while all
other tool selection remains model-native.
`agent.sh` now provides Linux-style `/help [COMMAND]` output. The full page
documents every local command, aliases, permission scope and numeric ranges;
topic help such as `/help max` shows the detailed syntax and effective limits.

## Agent routing and context profiles — 2026-08-11

The application now separates local commands, deterministic direct tools,
tool-then-model work and model-only requests. An unambiguous directory listing
uses `DIRECT_TOOL`, displays the typed tool result and explicitly reports
`model skipped`; no tool schema, prompt replay or MiniCPM generation is spent
on that request. A board measurement recorded `4.3 ms` tool time (`12.2 ms`
resident request time); the surrounding one-shot process still spent about
`10.8 s` loading the three model handles. Reports now expose route mode/reason,
route time, tool time, whether the model ran and total time.

Runtime profiles bind context, models, capabilities, generation limits and
numeric policy. The final matrix is ctx128 Chat-only, ctx1024 Chat+Agent,
ctx4096 Chat+Agent and ctx8192 Chat+Agent. Only ctx1024 is qualified in this
release. The other profiles are fail-closed `pending` declarations until their
exact OM sets pass descriptor, public-output cosine strictly greater than
`0.98`, greedy-token and board gates. Agent mode rejects ctx128 before loading
any model. At startup, the runtime also checks that mask/RoPE/K/V descriptor
geometry exactly matches the selected context.

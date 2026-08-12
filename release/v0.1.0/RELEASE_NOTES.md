# pico-minicpm5 v0.1.0

[中文](RELEASE_NOTES.zh-CN.md)

This source release captures the reproducible MiniCPM5-1B → ONNX → packed
24-layer PICO OM workflow and the accepted Hi3403 ctx1024 three-handle artifact
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
packs C4 embeddings directly and prepares RoPE sparsely. Hi3403 throughput rises
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
on that request. The same fail-closed route now covers current-directory,
literal file-window, literal-search and Git-status requests; any summary,
explanation, transformation or mutation falls back to MiniCPM. A board
measurement recorded `4.3 ms` tool time (`12.2 ms`
resident request time); the surrounding one-shot process still spent about
`10.8 s` loading the three model handles. Reports now expose route mode/reason,
route time, tool time, whether the model ran and total time.

Model-native turns now disclose read-only, write and shell schema groups only
as required by conservative intent rules. Ambiguous development tasks retain
the complete set. Successful tool responses are typed and referenceable, with
bounded output, truncation metadata and `read_result_page` continuation over
the most recent 16 results.

Runtime profiles bind context, models, capabilities, generation limits and
numeric policy. The final matrix is ctx128 Chat-only, ctx1024 Chat+Agent,
ctx4096 Chat+Agent and ctx8192 Chat+Agent. Only ctx1024 is qualified in this
release. The other profiles are fail-closed `pending` declarations until their
exact OM sets pass descriptor, public-output cosine strictly greater than
`0.98`, greedy-token and board gates. Agent mode rejects ctx128 before loading
any model. At startup, the runtime also checks that mask/RoPE/K/V descriptor
geometry exactly matches the selected context.

Cold startup now launches the executor before parsing the tokenizer, allowing
the 1.58 GB three-OM load and roughly 3-second tokenizer parse to overlap.
Three Hi3403 measurements improved from `10.8–11.5 s` to `8.2–8.6 s`.
Qualified profiles now also bind K/V/hidden output slots, removing the former
four-execute KV-identification probe. A ctx1024 end-to-end generation smoke
loaded in `7.4 s` and produced the accepted EOS sequence for `1+1 equals`.
Legacy invocations without a trusted slot contract keep the probe as a
compatibility fallback. Startup failures terminate the executor instead of
leaving a resident child behind.

Agent fixed system/tool-prefix K/V snapshots are now enabled by default after
a Hi3403 token-exact A/B. Restoring a 137-token prefix took `1.76 ms`; the same
32-token response remained byte-for-byte identical while request time fell
from `26.97 s` to `12.56 s` (`53.4%`). Set
`FIXED_PREFIX_SNAPSHOTS=0 ./app/agent.sh` for a full-replay diagnostic.

Deterministic context rebase also passed its Hi3403 long-session board gate.
Two runs each compacted 12 old direct-tool turns from `2808` to `810` prompt
tokens and emitted the exact same `[18655, 4569, EOS]` response. The repeat
restored a 643-token resident prefix in `7.15 ms`, reducing total latency from
`69.45 s` to `14.61 s` (`4.75x`).

Prompt ingestion no longer runs the vocabulary head and argmax for predictions
that a known next input token will discard. The final prompt position and every
generated position retain the complete transformer/head path. The same
token-exact board A/B skipped 809/812 and 166/169 head executions, reducing
cold latency from `86.70 s` to `69.45 s` (`19.89%`) and the resident-prefix
repeat from `18.17 s` to `14.61 s` (`19.59%`).

The source runtime now also contains a fail-closed planner for the future
`S128 -> S32 -> S16 -> strict S1 tail` native-prefill route. It records the
plan in request reports but deliberately enables only S1 in this qualified
bundle. Wider families require separate context-specific numeric, handoff and
Hi3403 board qualification before activation.

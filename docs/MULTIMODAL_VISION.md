# Two models on one NPU: the vision skill

[中文](MULTIMODAL_VISION.zh-CN.md)

Status: board-proven. `app/src/vision_jobs.py`, `app/src/vision_worker.py` and
`app/src/minicpm4v_vision.py` are the implementation; `describe_image` and
`--vision-queue` are the runtime surface.

## 1. Why a queue and not a call

MiniCPM5-1B answers questions. MiniCPM-4v-0.5B reads images. Each holds three
resident OM handles, and there is one NPU. A board agent that called the vision
pipeline inline would stop answering for as long as an image took — and an
image takes `21.5 s` here, against `3.2 s` for a greeting.

So the two are joined by a job queue rather than a function call.
`describe_image` writes a job and returns a job id in about a millisecond; a
separate `vision_worker` process claims it and looks; the answer is surfaced on
a later turn. The language model never blocks on the camera.

The queue is a directory of JSON files and nothing else — no daemon to keep
alive, no socket to bind. A job moves `queued → claimed → done|failed` by
atomic rename, so a crashed worker leaves a claimed job that can be seen and
requeued rather than a lost one, and either process can restart under the
other.

## 2. What the hardware refuses, and the way around it

Four handles are published. Three load. `decode.om` declares 53 inputs and 49
outputs — five, plus one port per layer for each of K and V — and this SDK caps
a model at 32 ports, so it is refused at load time. There is no KV-cached
decode step available on this board at all.

That looks fatal for generation and is not. `prefill_decode.om` emits logits
for its whole 200-row window, so the next token can be read off the last real
position — and the token after that comes from running prefill again with the
first one appended. Generation becomes repeated prefill:

```text
vision.om ──▶ resample.om ──▶ 64 vision tokens
                                   │
              ┌────────────────────┘
              ▼
     prefill_decode.om  ◀── PRE(9) + image(64) + MID(3) + question + POST(6) + answer-so-far
              │
              └──▶ logits[prefill_len - 1] ──▶ argmax ──▶ append, repeat
```

The window is 200 rows and a question is short, so roughly a hundred tokens of
answer fit. A caller that fills it is told, not truncated — silently dropping
the newest token would loop forever.

The cost is one full 200-row prefill per token. That is `0.52 s`, flat, and it
is not going down. It is why the queue exists.

## 3. Measured

Submit to done, a 1440×900 screenshot, 40-token cap, Hi3403:

| Stage | |
|---|---|
| claimed by the worker | `1.02 s` |
| first token visible | `1.98 s` |
| cadence | `0.52 s` per token, flat |
| done | `22.56 s` |

The `1.02 s` claim is the worker's poll interval (`--poll-seconds`), not model
time. Preprocess, `vision.om` and `resample.om` together are under a second and
are paid once per image, not once per token.

What it produced, unedited:

```text
这张图片展示了一个名为"HISpark"的软件界面。在顶部，可以看到一个名为"Model"的选项，
并且有一个"+"号按钮，这可能用于添加或创建…（达到 40 词上限）
```

## 4. The answer arrives as it is written

Delivering at turn boundaries is not the same as streaming, and the first
version was not streaming. Two gaps, both closed:

`report_vision` ran only after `input()` returned, so a user who asked about a
picture and then sat still saw nothing — the REPL was blocked in a read that
finishing could not interrupt. The prompt now polls: `select` on stdin with a
short timeout, and between wakeups a progress line redrawn in place. Without a
tty there is nothing to animate, so pipes, tests and recordings still get an
ordinary read.

And the worker wrote once, at the end — `21.5 s` of silence followed by a
paragraph. `VisionQueue.progress` publishes after every token. The record stays
`claimed`, so it is a live view and not a delivery: `collect()` still ignores
it, and `finish()` clears it so nothing is shown twice.

Two windows still do not refresh, and both are deliberate: the vision line is
not redrawn while the language model is streaming its own reply, and polling
yields to `input()` once the user has typed the first character.

## 5. Disclosure is not free

A tool schema is a fixed prompt prefix, charged in prefill tokens on every turn
that names it. `describe_image` therefore gets a one-tool profile of its own
rather than joining the read set, and it is disclosed only when two things hold
at once: the turn has vision intent *and* names a file. `看看这个目录` and
`看看这张图` with no filename both stay on the read set.

It is also disclosed only where a worker exists. Without `--vision-queue` the
tool could only ever be called and refused, and the schema would be charged
anyway, so those turns fall back to the read set — which can at least say what
the file is.

## 6. Running it

The worker owns the 4v handles and nothing else:

```sh
# $QUEUE is any directory both processes can write; $VLM holds the three 4v
# handles plus tokenizer.json and token_emb.bin; $EXE is the persistent ACL
# executor built by docs/EXECUTOR_BUILD.md.
python3 -u src/vision_worker.py \
  --queue "$QUEUE" \
  --model-dir "$VLM" \
  --executable "$EXE" \
  --library-path /opt/lib/svp_npu --library-path /opt/lib --library-path /opt/lib/npu \
  --poll-seconds 1.0 --max-new 40
```

The agent is pointed at the same directory:

```sh
./agent.sh --vision-queue "$QUEUE"
```

`--max-new` is a latency budget, not a quality knob: every token is a whole
prefill, so 40 tokens is 21 seconds and 80 is 42.

## 7. Preprocessing contract

`minicpm4v_vision.py` ports the published C++ contract rather than guessing at
it: 512×512 CUBIC, `(x/255 − 0.5)/0.5`, the 16×16 patch reshape as a five-axis
transpose, the 200-token template and its two-pass attention mask, and a greedy
longest-match vocabulary — the table is **not** BPE. `MASK_MIN_VALUE` is
`-9999999.0`, read from the header rather than assumed to be `-10000`. The
300 MB embedding table is read by `seek` and never loaded.

Two wire details are worth repeating because both produced silent corruption
before they were pinned: the executor writes **every output size first, then
every payload** — reading them interleaved parses the second size out of the
first tensor's bytes — and `public_inputs` in the request header is the model's
public input-port count, not the number of writes.

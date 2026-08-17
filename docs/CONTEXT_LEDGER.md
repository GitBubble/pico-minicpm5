# The context ledger

[中文](CONTEXT_LEDGER.zh-CN.md)

Status: accepted design; `app/src/context_ledger.py` is the implementation.

## 1. Why

A prompt token costs `79.49 ms` to ingest on the qualified ctx1024 board, and
the whole conversation — system prose, tool schemas, history, tool results and
the turn being asked — has to fit inside 1024 of them. Both limits are hard, and
until now neither was observable at runtime. The agent reported how many prompt
tokens a request executed, but not what they were spent on.

The first time the question was asked directly, the answer was not the one
anyone would have guessed. Reconstructing the shape of a measured board turn —
a directory listing already in history, and the user saying hello:

| segment | tokens | share | ingest |
|---|---:|---:|---:|
| tool schema, 8 tools | 481 | 61.8% | 38.23 s |
| system prose | 193 | 24.8% | 15.35 s |
| the tool result in history | 80 | 10.3% | 6.36 s |
| the conversation itself | 24 | 3.1% | 1.92 s |
| `<s>` | 1 | 0.1% | 0.08 s |

Two thirds of the context, and thirty-eight seconds of latency, went to
describing tools to a model that was being asked to say hello. That is the
defect the ledger exists to make visible, and it is not visible any other way:
every number above is correct, none of them was in any log.

## 2. Exact, not estimated

Agent harnesses that talk to a model over an API have to estimate token counts,
because the tokenizer is on the other side of the network. We run the tokenizer
in-process, and the runtime already encodes the assembled prompt once in order
to feed it to the model. So attribution can be exact, and it can be free.

The mechanism is offset attribution. The tokenizer returns one `(start, end)`
character pair per token; each token is assigned to the segment that holds its
**first** character. Encoding each segment separately would be both slower and
wrong — the tokenizer merges across a segment boundary, so the parts do not sum
to the whole.

A token whose span crosses a boundary is counted once, in the segment where it
starts, and the number of such tokens is reported as `boundary_tokens`. It is
the only ambiguity in the accounting, so it is measured rather than hidden.

## 3. Segmentation

Segments are read back out of the rendered prompt along the MiniCPM5 wire
format, not handed down from the code that built it. That keeps the ledger
independent of the renderer — including a renderer written in another language —
and makes the segment table checkable: the segments must tile the string
exactly, and `measure` reports any character that no segment claims.

```text
preamble           the leading <s>
system             the system message, minus the tool block
tool_schema        the <tools> … </tools> block, labelled by the tools in it
history_user       an earlier user turn
history_assistant  an earlier assistant turn
tool_result        a <tool_response> block
current_user       the turn being asked
generation_prompt  the trailing assistant prefix
```

## 4. Cost and pressure

The resident K/V prefix is always a prefix by token position, so the split
between tokens already in the cache and tokens that must be ingested is exact
rather than apportioned. Given the profile's measured `prompt_token_ms`, each
segment reports the seconds its *new* tokens will cost. A segment that is
entirely resident costs nothing, which is what makes fixed-prefix snapshots
worth having and what makes prefix churn expensive.

Pressure is `total_tokens / (capacity - reserve_tokens)`. It crosses 1.0 before
the model refuses anything, so it is the signal that a rebase is due.

## 5. Contract

The record carries `schema: pico.minicpm5.context-ledger.v1`. Three laws hold
for every record, and each has a test:

1. **Conservation.** The segment token counts sum to `total_tokens`. Anything
   no segment claimed is one of those rows, kind `unattributed`, and its count
   is repeated in the scalar `unattributed_tokens` so a reader checking the
   total never has to add it twice.
2. **Resident split.** The segment `new_tokens` sum to
   `total_tokens - resident_tokens`.
3. **Tiling.** Segments never overlap and never end past the text. A table that
   breaks either rule raises `LedgerError` rather than producing a plausible
   wrong number.

## 6. Portability

The framework this design borrows from meters tokens through a service graph.
We deliberately do not. `context_ledger.py` imports nothing but `dataclasses`
and `re`: not the tokenizer, not the agent, not the runtime. Its input is a
string, a segment table and one offset pair per token; its output is a plain
dictionary.

The scan is a single pass over the offsets with a cursor into the sorted segment
table — `O(n_tokens + n_segments)`, no allocation per token. The intended path
is a Rust implementation of the same pass, gated the way every other component
here is gated: both implementations emit a `pico.minicpm5.context-ledger.v1`
record for the same input, and the records must match. The schema is the
contract; the tests are the conformance suite.

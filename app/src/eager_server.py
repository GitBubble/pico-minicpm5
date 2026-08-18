#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Session-side eager-prefill protocol for the merged board server.

``merged_board_server.Merged`` INHERITS ``EagerSessionMixin`` directly; there
is no runtime subclass factory, no path-loaded module and no import-time
monkeypatch.  (Run 1 of the board A/B lost an entire arm to a staged overlay
that ``sys.path[0]`` shadowing bound to a stale copy -- that layout is
deliberately not repeated here.)

The mixin only adds four methods and touches no state the base class does not
already own:

    resident_tokens() -> tuple      host view of the device-resident prefix
    prefill_extend(tokens) -> None  append S1 rows at the resident length
    rewind_to(n) -> None            truncate host metadata (device rows stay
                                    causally masked, same contract as
                                    reset_prefix_cache)
    reset_kv() -> None              full invalidation (fail-closed path)

``prefill_extend`` mirrors the resident S1 prompt-ingestion branch of
``Merged.generate`` -- same mask (built by the SHARED
``_attention_mask_bytes`` helper, so the two paths cannot drift), same writes
tuple, same ``publish=False`` execute, same position-addressed
``_scatter_kv`` -- with the 200 MB vocabulary head permanently skipped: an
eager row always has a known successor, so its prediction would be discarded.

Constraints inherited from the server generation:
* resident KV only (host_kv mirrors are a diagnostic path);
* S1 rows only -- wide (S16/S128) blocks stay owned by the runtime registry
  inside ``generate``; re-planning a wide schedule under rollback is future
  work;
* fixed-prefix snapshots compose: ``_prepare_prefix_snapshot`` only restores
  when the resident prefix is SHORTER than the fixed prefix, which cannot
  happen after eager feeding past it.
"""
from __future__ import annotations

import time

from eager_prefill import EagerCapacityReached


class EagerSessionMixin:
    """The four-method eager session protocol, on the real ``Merged``."""

    def resident_tokens(self):
        return tuple(self._resident_tokens)

    def rewind_to(self, count):
        if count < 0 or count > len(self._resident_tokens):
            raise ValueError("rewind target outside resident prefix")
        # Position-addressed scatter: rows beyond the host prefix stay in the
        # device cache but are causally masked (same contract as
        # reset_prefix_cache) and are overwritten by the next prefill at
        # those positions.
        del self._resident_tokens[count:]

    def reset_kv(self):
        self.reset_prefix_cache()

    def prefill_extend(self, tokens):
        """Append S1 rows at the resident length; vocabulary head skipped."""
        if not self.resident_kv:
            raise RuntimeError("eager prefill requires resident KV")
        self._require_live_wide_session()
        for token in tokens:
            position = len(self._resident_tokens)
            if position >= self.past:
                # Every eager row must scatter into the context-1 cache rows;
                # the final legal position is generate's business.
                raise EagerCapacityReached(position)
            self._deadline = time.monotonic() + self.timeout
            model = (self.prefill_index if position == 0
                     else self.decode_index)
            desc_in = self.descriptors[model][0]
            mask_width = (self.prefill_context if position == 0
                          else self.context)
            writes = [
                (0, 0, self._hidden_input(token, desc_in[0])),
                (1, 0, self._attention_mask_bytes(position, mask_width)),
                (2, 0, self._rope_matrix_bytes(position)),
            ]
            self._run(model, writes, publish=False)
            self._scatter_kv(model, position)
            self._resident_tokens.append(int(token))

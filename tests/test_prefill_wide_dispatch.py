from __future__ import annotations

import importlib.util
import hashlib
import io
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest


PROJECT = Path(__file__).resolve().parents[1]
APP_SRC = PROJECT / "app" / "src"


def _server_module():
    sys.path.insert(0, str(APP_SRC))
    spec = importlib.util.spec_from_file_location(
        "pico_minicpm5_wide_dispatch_test",
        APP_SRC / "merged_board_server.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _handler(runtime, model: Path, *, context=128, width=16, model_index=1,
             mask_negative=None):
    embedding_row_bytes = runtime.HIDDEN * 4
    cache_bytes = runtime.CHANNELS * (context - 1) * runtime.ROW_F16_BYTES
    publisher = runtime.WidePublisherABI(
        width=width,
        k_output_slot=0,
        v_output_slot=1,
        hidden_output_slot=2,
        hidden_output_bytes=embedding_row_bytes,
    )
    descriptor = runtime.WideHandleDescriptor(
        input_bytes=(
            width * embedding_row_bytes,
            width * context * 4,
            width * runtime.HEAD_DIM * runtime.HEAD_DIM * 4,
            cache_bytes,
            cache_bytes,
        ),
        output_bytes=(
            publisher.kv_output_bytes,
            publisher.kv_output_bytes,
            publisher.hidden_output_bytes,
        ),
        ready_descriptor_sha256="a" * 64,
    )
    if mask_negative is None:
        mask_negative = runtime.MASK_NEGATIVE
    return runtime.WidePrefillHandler(runtime.WideHandleSpec(
        width=width,
        context=context,
        model_index=model_index,
        model_path=model,
        descriptor=descriptor,
        publisher=publisher,
        embedding_row_bytes=embedding_row_bytes,
        mask_negative=mask_negative,
    ))


def test_wide_handler_rejects_unqualified_mask_negative(
        tmp_path: Path) -> None:
    server = _server_module()
    runtime = server.prefill_runtime_contract

    with pytest.raises(
            runtime.PrefillRuntimeError,
            match="mask negative value does not match qualification"):
        _handler(
            runtime, tmp_path / "s16.om", mask_negative=-1e-9)


def test_wide_handler_rejects_kv_output_slot_swap(tmp_path: Path) -> None:
    server = _server_module()
    runtime = server.prefill_runtime_contract

    with pytest.raises(
            runtime.PrefillRuntimeError,
            match="publisher slots must be exactly"):
        runtime.WidePublisherABI(
            width=16,
            k_output_slot=1,
            v_output_slot=0,
            hidden_output_slot=2,
            hidden_output_bytes=runtime.HIDDEN * 4,
        )


def test_wide_handler_rejects_kv_input_slot_swap(tmp_path: Path) -> None:
    server = _server_module()
    runtime = server.prefill_runtime_contract
    handler = _handler(runtime, tmp_path / "s16.om")
    spec = handler.spec

    with pytest.raises(
            runtime.PrefillRuntimeError,
            match="input slots must be exactly"):
        runtime.WideHandleSpec(
            width=spec.width,
            context=spec.context,
            model_index=spec.model_index,
            model_path=spec.model_path,
            descriptor=spec.descriptor,
            publisher=spec.publisher,
            embedding_row_bytes=spec.embedding_row_bytes,
            k_cache_input_slot=4,
            v_cache_input_slot=3,
        )


class _TraceTransport:
    """Byte-addressed fake for the typed wide transport boundary."""

    def __init__(self, runtime, handler, *, fail_at=None):
        self.runtime = runtime
        self.spec = handler.spec
        self.fail_at = fail_at
        self.trace = []
        self.writes = ()
        self.discarded = False
        self.past = self.spec.context - 1
        self.row = runtime.ROW_F16_BYTES
        self.stride = self.past * self.row
        cache_bytes = runtime.CHANNELS * self.stride
        self.canonical = {
            3: bytearray((index * 17 + 3) & 0xFF
                         for index in range(cache_bytes)),
            4: bytearray((index * 29 + 5) & 0xFF
                         for index in range(cache_bytes)),
        }
        self.wide = {
            3: bytearray(b"\xcc" * cache_bytes),
            4: bytearray(b"\xdd" * cache_bytes),
        }

    def _fail(self, stage):
        if self.fail_at == stage:
            raise RuntimeError(f"injected {stage} failure")

    def wide_copy_prefix(self, *, source_model, destination_model,
                         token_count):
        self.trace.append((
            "copy-prefix", source_model, destination_model, token_count))
        self._fail("copy-prefix")
        length = token_count * self.row
        for slot in (3, 4):
            for channel in range(self.runtime.CHANNELS):
                offset = channel * self.stride
                self.wide[slot][offset:offset + length] = \
                    self.canonical[slot][offset:offset + length]

    def wide_execute(self, *, spec, writes):
        self.trace.append(("execute", spec.model_index, len(writes)))
        self.writes = writes
        self._fail("execute")

    def wide_publish_kv(self, *, spec, start):
        self.trace.append(("publish", spec.model_index, start, spec.width))
        length = spec.width * self.row
        for slot, marker in ((3, 0x31), (4, 0x72)):
            for channel in range(self.runtime.CHANNELS):
                offset = channel * self.stride + start * self.row
                self.canonical[slot][offset:offset + length] = \
                    bytes([(marker + channel) & 0xFF]) * length
                # A publish failure is allowed to occur after a partial device
                # mutation. The whole session must still be discarded.
                if self.fail_at == "publish" and channel == 0:
                    raise RuntimeError("injected publish failure")

    def wide_discard_session(self, *, spec, stage, error):
        self.trace.append(("discard", spec.model_index, stage, str(error)))
        self.discarded = True


def _embedding_row(token, want):
    return struct.pack("<I", token) * (want // 4)


def _rope_row(position):
    size = 128 * 128 * 4
    return struct.pack("<I", position) * (size // 4)


def _file(root: Path, name: str, payload: bytes) -> Path:
    path = root / name
    path.write_bytes(payload)
    return path.resolve()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_live_wide_startup_rehashes_complete_route_before_spawn(
        tmp_path: Path) -> None:
    server = _server_module()
    runtime = server.prefill_runtime_contract
    bootstrap = _file(tmp_path, "bootstrap.om", b"bootstrap")
    decode = _file(tmp_path, "decode.om", b"decode")
    executor = _file(tmp_path, "executor", b"executor")
    runner = _file(tmp_path, "runner.py", b"runner")
    head = _file(tmp_path, "head.om", b"head")
    embedding = _file(tmp_path, "embedding.bin", b"embedding")
    build = _file(tmp_path, "build.json", b"{}")
    bootstrap_descriptor = _file(
        tmp_path, "bootstrap-descriptor.bin", b"bootstrap-descriptor")
    decode_descriptor = _file(
        tmp_path, "decode-descriptor.bin", b"decode-descriptor")
    qualification = _file(tmp_path, "s1-q.json", b"{}")
    wide_model = _file(tmp_path, "s16.om", b"wide-model")
    handler = _handler(runtime, wide_model)
    identity = runtime.activation_contract.StrictS1Identity(
        qualification_sha256=_digest(qualification),
        bootstrap_om_sha256=_digest(bootstrap),
        canonical_decode_om_sha256=_digest(decode),
        head_om_sha256=_digest(head),
        embedding_sha256=_digest(embedding),
        build_manifest_sha256=_digest(build),
        runner_sha256=_digest(runner),
        executor_sha256=_digest(executor),
        bootstrap_ready_descriptor_sha256=_digest(bootstrap_descriptor),
        canonical_ready_descriptor_sha256=_digest(decode_descriptor),
        resident_bytes=1,
    )
    anchor = runtime.activation_contract.StrictS1Anchor(
        qualification=qualification,
        bootstrap_model=bootstrap,
        canonical_decode_model=decode,
        head_model=head,
        embedding=embedding,
        build_manifest=build,
        runner=runner,
        executor=executor,
        bootstrap_ready_descriptor=bootstrap_descriptor,
        canonical_ready_descriptor=decode_descriptor,
        identity=identity,
        declaration={},
    )
    s1_only_activation = runtime.PrefillRuntimeRegistry(
        context=128,
        activation_manifest=tmp_path / "activation.json",
        activation_report={},
        qualified_widths=(1,),
        enabled_widths=(1,),
        handler_widths=(),
        unavailable={},
        handlers=(),
        strict_s1_anchor=anchor,
    )
    s1_only_activation.validate_live_startup_identity(
        executable=executor, decode=decode, prefill=bootstrap, head=head,
        embedding=embedding, runner=runner)
    decode.write_bytes(b"replaced-no-handler-decode")
    with pytest.raises(
            runtime.PrefillRuntimeError,
            match="canonical decode OM SHA-256"):
        s1_only_activation.validate_live_startup_identity(
            executable=executor, decode=decode, prefill=bootstrap, head=head,
            embedding=embedding, runner=runner)
    decode.write_bytes(b"decode")

    registry = runtime.PrefillRuntimeRegistry(
        context=128,
        activation_manifest=tmp_path / "activation.json",
        activation_report={},
        qualified_widths=(16, 1),
        enabled_widths=(16, 1),
        handler_widths=(16,),
        unavailable={},
        handlers=(handler,),
        strict_s1_anchor=anchor,
        activated_model_sha256=((16, _digest(wide_model)),),
    )

    registry.validate_live_startup_identity(
        executable=executor, decode=decode, prefill=bootstrap, head=head,
        embedding=embedding, runner=runner)
    with pytest.raises(runtime.PrefillRuntimeError, match="bootstrap OM path"):
        registry.validate_live_startup_identity(
            executable=executor, decode=decode, prefill=decode, head=head,
            embedding=embedding, runner=runner)

    head.write_bytes(b"replaced-head")
    with pytest.raises(runtime.PrefillRuntimeError, match="head OM SHA-256"):
        registry.validate_live_startup_identity(
            executable=executor, decode=decode, prefill=bootstrap, head=head,
            embedding=embedding, runner=runner)
    head.write_bytes(b"head")

    embedding.write_bytes(b"replaced-embedding")
    with pytest.raises(runtime.PrefillRuntimeError, match="embedding SHA-256"):
        registry.validate_live_startup_identity(
            executable=executor, decode=decode, prefill=bootstrap, head=head,
            embedding=embedding, runner=runner)
    embedding.write_bytes(b"embedding")

    wide_model.write_bytes(b"replaced-after-activation")
    with pytest.raises(runtime.PrefillRuntimeError, match="S16 model SHA-256"):
        registry.validate_live_startup_identity(
            executable=executor, decode=decode, prefill=bootstrap, head=head,
            embedding=embedding, runner=runner)


def test_wide_handler_prepares_exact_inputs_and_orders_transport_calls(
        tmp_path: Path) -> None:
    server = _server_module()
    runtime = server.prefill_runtime_contract
    handler = _handler(runtime, tmp_path / "s16.om")
    transport = _TraceTransport(runtime, handler)
    start = 7
    tokens = tuple(range(101, 117))

    result = handler.execute(
        transport=transport,
        canonical_model=0,
        start=start,
        token_ids=tokens,
        embedding_row=_embedding_row,
        rope_row=_rope_row,
    )

    assert transport.trace == [
        ("copy-prefix", 0, 1, start),
        ("execute", 1, 3),
        ("publish", 1, start, 16),
    ]
    assert [(write.input_slot, write.offset) for write in transport.writes] == [
        (0, 0), (1, 0), (2, 0)]
    embeddings, masks, ropes = (
        write.payload for write in transport.writes)
    row_bytes = runtime.HIDDEN * 4
    assert len(embeddings) == 16 * row_bytes
    assert embeddings[:row_bytes] == _embedding_row(tokens[0], row_bytes)
    assert embeddings[-row_bytes:] == _embedding_row(tokens[-1], row_bytes)
    assert len(masks) == 16 * 128 * 4
    for row_index, position in ((0, start), (15, start + 15)):
        values = struct.unpack_from("<128f", masks, row_index * 128 * 4)
        assert values[:position] == (0.0,) * position
        assert values[position:127] == \
            (handler.spec.mask_negative,) * (127 - position)
        assert values[127] == 0.0
    rope_bytes = 128 * 128 * 4
    assert len(ropes) == 16 * rope_bytes
    assert ropes[:rope_bytes] == _rope_row(start)
    assert ropes[-rope_bytes:] == _rope_row(start + 15)
    assert result.width == 16 and result.start == start and result.stop == 23
    assert result.hidden == runtime.ResidentHidden(
        model_index=1, output_slot=2, byte_size=row_bytes, position=22)


def test_wide_copy_and_publish_touch_only_declared_kv_rows(
        tmp_path: Path) -> None:
    server = _server_module()
    runtime = server.prefill_runtime_contract
    handler = _handler(runtime, tmp_path / "s16.om")
    transport = _TraceTransport(runtime, handler)
    start = 9
    before = {slot: bytes(value)
              for slot, value in transport.canonical.items()}

    handler.execute(
        transport=transport,
        canonical_model=0,
        start=start,
        token_ids=tuple(range(16)),
        embedding_row=_embedding_row,
        rope_row=_rope_row,
    )

    stop = start + 16
    for slot, wide_sentinel in ((3, 0xCC), (4, 0xDD)):
        for channel in range(runtime.CHANNELS):
            base = channel * transport.stride
            prefix = start * transport.row
            end = stop * transport.row
            assert transport.wide[slot][base:base + prefix] == \
                before[slot][base:base + prefix]
            assert transport.wide[slot][base + prefix:base + transport.stride] == \
                bytes([wide_sentinel]) * (transport.stride - prefix)
            assert transport.canonical[slot][base:base + prefix] == \
                before[slot][base:base + prefix]
            assert transport.canonical[slot][base + end:base + transport.stride] == \
                before[slot][base + end:base + transport.stride]
            assert transport.canonical[slot][base + prefix:base + end] != \
                before[slot][base + prefix:base + end]


@pytest.mark.parametrize(
    ("failure,expected_stage"),
    (("copy-prefix", "copy-prefix"),
     ("execute", "execute"),
     ("publish", "publish-kv")),
)
def test_wide_failure_discards_session_and_requires_rebuild(
        tmp_path: Path, failure: str, expected_stage: str) -> None:
    server = _server_module()
    runtime = server.prefill_runtime_contract
    handler = _handler(runtime, tmp_path / "s16.om")
    transport = _TraceTransport(runtime, handler, fail_at=failure)

    with pytest.raises(runtime.WideBlockExecutionError) as raised:
        handler.execute(
            transport=transport,
            canonical_model=0,
            start=3,
            token_ids=tuple(range(16)),
            embedding_row=_embedding_row,
            rope_row=_rope_row,
        )

    assert raised.value.stage == expected_stage
    assert raised.value.requires_rebuild is True
    assert transport.discarded is True
    assert transport.trace[-1][0:3] == ("discard", 1, expected_stage)
    if failure == "execute":
        assert not any(event[0] == "publish" for event in transport.trace)


def test_wide_prepare_failure_also_discards_session(tmp_path: Path) -> None:
    server = _server_module()
    runtime = server.prefill_runtime_contract
    handler = _handler(runtime, tmp_path / "s16.om")
    transport = _TraceTransport(runtime, handler)

    with pytest.raises(runtime.WideBlockExecutionError) as raised:
        handler.execute(
            transport=transport,
            canonical_model=0,
            start=3,
            token_ids=tuple(range(16)),
            embedding_row=lambda _token, _want: b"short",
            rope_row=_rope_row,
        )

    assert raised.value.stage == "prepare-inputs"
    assert transport.trace[-1][0:3] == ("discard", 1, "prepare-inputs")
    assert transport.discarded is True


def test_wide_preflight_rejects_unsupported_or_terminal_block_without_transport(
        tmp_path: Path) -> None:
    server = _server_module()
    runtime = server.prefill_runtime_contract
    handler = _handler(runtime, tmp_path / "s16.om")
    transport = _TraceTransport(runtime, handler)
    registry = runtime.strict_s1_runtime(128)
    forged = runtime.schedule_contract.plan_prefill(
        1, 17, context=128, enabled_widths=(16, 1))

    with pytest.raises(runtime.PrefillRuntimeError, match="no registered"):
        registry.require_executable(forged)
    with pytest.raises(runtime.PrefillRuntimeError, match="context-1"):
        handler.execute(
            transport=transport,
            canonical_model=0,
            start=112,
            token_ids=tuple(range(16)),
            embedding_row=_embedding_row,
            rope_row=_rope_row,
        )
    assert transport.trace == []


def _runtime_registry(runtime, handler):
    return runtime.PrefillRuntimeRegistry(
        context=handler.spec.context,
        activation_manifest=Path("fake-activation.json"),
        activation_report={"schema": "fake.activation.v4"},
        qualified_widths=(handler.spec.width, 1),
        enabled_widths=(handler.spec.width, 1),
        handler_widths=(handler.spec.width,),
        unavailable={},
        handlers=(handler,),
    )


def test_runtime_plan_forces_terminal_context_position_to_s1(
        tmp_path: Path) -> None:
    server = _server_module()
    runtime = server.prefill_runtime_contract
    handler = _handler(runtime, tmp_path / "s16.om")
    registry = _runtime_registry(runtime, handler)

    schedule = registry.plan(0, 128)

    assert [(segment.width, segment.count, segment.start, segment.stop)
            for segment in schedule.segments] == [
        (1, 1, 0, 1),
        (16, 7, 1, 113),
        (1, 14, 113, 127),
        (1, 1, 127, 128),
    ]
    assert all(segment.stop <= 127 for segment in schedule.segments
               if segment.width > 1)


def test_activation_only_enables_an_exact_registered_live_handler(
        monkeypatch, tmp_path: Path) -> None:
    server = _server_module()
    runtime = server.prefill_runtime_contract
    model = tmp_path / "models" / "s16.om"
    model.parent.mkdir()
    model.write_bytes(b"qualified fake model identity")
    handler = _handler(runtime, model)

    class Activation:
        enabled_widths = (16, 1)
        disabled = {"S128": "not admitted", "S32": "not admitted"}
        strict_s1 = object()
        blocks = (SimpleNamespace(
            width=16,
            model=model.resolve(),
            model_sha256=_digest(model),
            ready_descriptor_sha256=(
                handler.spec.descriptor.ready_descriptor_sha256),
        ),)

        def to_dict(self):
            return {"schema": "fake.activation.v4", "enabled_widths": [16, 1]}

    monkeypatch.setattr(
        runtime.activation_contract, "load_activation",
        lambda *_args, **_kwargs: Activation())
    registry = runtime.load_runtime_registry(
        activation_manifest=tmp_path / "activation.json",
        deployment_root=tmp_path,
        context=128,
        available_bytes=900,
        base_resident_bytes=400,
        reserve_bytes=200,
        handlers=(handler,),
    )

    assert registry.enabled_widths == (16, 1)
    assert registry.handlers == (handler,)
    assert registry.handler_for(16) is handler
    registry.validate_loaded_handlers(
        [([], []),
         (handler.spec.descriptor.input_bytes,
          handler.spec.descriptor.output_bytes),
         ([], [])],
        [Path("decode.om"), model, Path("head.om")],
    )
    drifted_outputs = list(handler.spec.descriptor.output_bytes)
    drifted_outputs[2] += 4
    with pytest.raises(runtime.PrefillRuntimeError, match="descriptor drift"):
        registry.validate_loaded_handlers(
            [([], []),
             (handler.spec.descriptor.input_bytes, drifted_outputs),
             ([], [])],
            [Path("decode.om"), model, Path("head.om")],
        )


def test_head_handoff_rejects_any_descriptor_truncation_before_frame() -> None:
    server = _server_module()
    runtime = server.prefill_runtime_contract
    session = object.__new__(server.Merged)
    session.models = [Path("wide.om"), Path("head.om")]
    session.head_index = 1
    session.descriptors = [
        ([], [1, 1, runtime.HIDDEN * 4]),
        ([runtime.HIDDEN * 16], [8]),
    ]
    session.process = SimpleNamespace(stdin=io.BytesIO())

    with pytest.raises(RuntimeError, match="exactly match"):
        session._predict_from_resident_hidden(runtime.ResidentHidden(
            model_index=0,
            output_slot=2,
            byte_size=runtime.HIDDEN * 4,
            position=16,
        ))

    assert session.process.stdin.getvalue() == b""


def _manual_session(server, handler):
    runtime = server.prefill_runtime_contract
    session = object.__new__(server.Merged)
    session.context = handler.spec.context
    session.past = session.context - 1
    session.timeout = 1.0
    session.row_f16 = runtime.ROW_F16_BYTES
    session.cache_bytes = runtime.CHANNELS * session.past * session.row_f16
    session.resident_kv = True
    session._resident_tokens = []
    session._prefix_snapshots = {}
    session._next_prefix_snapshot_id = 1
    session._wide_session_discarded = False
    session.last_prefix_metrics = session._empty_prefix_metrics()
    session.prefill_runtime = _runtime_registry(runtime, handler)
    session.last_prefill_runtime = session.prefill_runtime.to_dict()
    session.models = [Path("decode.om"), handler.spec.model_path, Path("head.om")]
    session.decode_index = 0
    session.prefill_index = 0
    session.head_index = 2
    s1_kv = runtime.CHANNELS * runtime.HEAD_DIM * 4
    session.descriptors = [
        (
            [runtime.HIDDEN * 4, session.context * 4,
             runtime.HEAD_DIM * runtime.HEAD_DIM * 4,
             session.cache_bytes, session.cache_bytes],
            [s1_kv, s1_kv, runtime.HIDDEN * 4],
        ),
        (list(handler.spec.descriptor.input_bytes),
         list(handler.spec.descriptor.output_bytes)),
        ([runtime.HIDDEN * 4], [8]),
    ]
    session.kv_slots = {0: (0, 1), 1: (0, 1)}
    session.hidden_slots = {0: 2, 1: 2}
    session.process = SimpleNamespace()
    session._hidden_input = _embedding_row
    session._rope_matrix_bytes = _rope_row
    trace = []
    s1_positions = []
    head_handoffs = []

    def run(model, writes, *args, **kwargs):
        rope = next(payload for slot, _offset, payload in writes if slot == 2)
        position = struct.unpack_from("<I", rope)[0]
        s1_positions.append(position)
        trace.append(("s1", model, position))
        return []

    session._run = run
    session._scatter_kv = lambda model, position: trace.append(
        ("s1-publish", model, position))
    session.wide_copy_prefix = lambda **kwargs: trace.append((
        "copy-prefix", kwargs["source_model"], kwargs["destination_model"],
        kwargs["token_count"]))
    session.wide_execute = lambda **kwargs: trace.append((
        "wide-execute", kwargs["spec"].model_index,
        tuple(write.input_slot for write in kwargs["writes"])))
    session.wide_publish_kv = lambda **kwargs: trace.append((
        "wide-publish", kwargs["spec"].model_index, kwargs["start"],
        kwargs["spec"].width))
    session.wide_discard_session = lambda **kwargs: trace.append((
        "discard", kwargs["stage"]))

    def predict(hidden):
        head_handoffs.append(hidden)
        trace.append((
            "head", hidden.model_index, hidden.output_slot, hidden.position))
        return 99, 0.25, 0.05

    session._predict_from_resident_hidden = predict
    return session, trace, s1_positions, head_handoffs


def test_generate_executes_s16_then_s1_tail_without_intermediate_heads(
        tmp_path: Path) -> None:
    server = _server_module()
    handler = _handler(
        server.prefill_runtime_contract, tmp_path / "s16.om")
    session, trace, s1_positions, head_handoffs = \
        _manual_session(server, handler)
    prompt = list(range(20))

    reason, ids, steps = session.generate(prompt, 1, set(), reuse_prefix=True)

    assert reason == "max" and ids == [99]
    assert len(steps) == len(prompt)
    assert s1_positions == [0, 17, 18, 19]
    assert [event[0] for event in trace] == [
        "s1", "s1-publish", "copy-prefix", "wide-execute", "wide-publish",
        "s1", "s1-publish", "s1", "s1-publish", "s1", "s1-publish",
        "head",
    ]
    assert [phase["prefill_width"] for phase in session.last_phase_steps] == [
        1, 16, 1, 1, 1]
    assert [phase["head_skipped"] for phase in session.last_phase_steps] == [
        True, True, True, True, False]
    assert session.last_phase_steps[1]["token_count"] == 16
    assert session.last_phase_steps[1]["stop"] == 17
    assert head_handoffs == [server.prefill_runtime_contract.ResidentHidden(
        model_index=0, output_slot=2,
        byte_size=server.prefill_runtime_contract.HIDDEN * 4, position=19)]
    assert session._resident_tokens == prompt


def test_generate_hands_final_wide_hidden_directly_to_head(
        tmp_path: Path) -> None:
    server = _server_module()
    runtime = server.prefill_runtime_contract
    handler = _handler(runtime, tmp_path / "s16.om")
    session, trace, s1_positions, head_handoffs = \
        _manual_session(server, handler)
    prompt = list(range(17))

    reason, ids, _steps = session.generate(
        prompt, 1, set(), reuse_prefix=True)

    assert reason == "max" and ids == [99]
    assert s1_positions == [0]
    assert head_handoffs == [runtime.ResidentHidden(
        model_index=1, output_slot=2,
        byte_size=runtime.HIDDEN * 4, position=16)]
    assert [phase["prefill_width"] for phase in session.last_phase_steps] == [
        1, 16]
    assert session.last_phase_steps[1]["hidden_handoff"] == {
        "model_index": 1, "output_slot": 2, "position": 16}
    assert [event[0] for event in trace].count("head") == 1
    assert not any(event[0] == "s1" and event[2] > 0 for event in trace)


def test_fixed_prefix_boundary_creates_snapshot_before_next_wide_block(
        tmp_path: Path) -> None:
    server = _server_module()
    runtime = server.prefill_runtime_contract
    handler = _handler(runtime, tmp_path / "s16.om")
    session, trace, s1_positions, _head_handoffs = \
        _manual_session(server, handler)
    saved = []
    session._save_input_snapshot = lambda snapshot_id, count: saved.append(
        (snapshot_id, count))
    prompt = list(range(36))
    fixed = tuple(prompt[:20])

    reason, ids, _steps = session.generate(
        prompt, 1, set(), reuse_prefix=True,
        prefix_snapshot_key="agent-fixed", prefix_snapshot_tokens=fixed)

    assert reason == "max" and ids == [99]
    assert saved == [(1, 20)]
    assert session._prefix_snapshots == {"agent-fixed": (1, fixed)}
    assert session.last_prefill_schedule["hard_boundaries"] == [20]
    assert s1_positions == [0, 17, 18, 19]
    wide_starts = [event[2] for event in trace
                   if event[0] == "wide-publish"]
    assert wide_starts == [1, 20]
    assert not any(start < 20 < start + 16 for start in wide_starts)


def test_fixed_and_terminal_boundaries_are_preflighted_together(
        tmp_path: Path) -> None:
    server = _server_module()
    runtime = server.prefill_runtime_contract
    handler = _handler(runtime, tmp_path / "s16.om")
    registry = _runtime_registry(runtime, handler)

    schedule = registry.plan(0, 128, hard_boundaries=(20,))

    assert schedule.hard_boundaries == (20, 127)
    assert all(not (segment.start < 20 < segment.stop)
               for segment in schedule.segments)
    assert all(segment.stop <= 127 for segment in schedule.segments
               if segment.width > 1)
    assert schedule.segments[-1].width == 1
    assert schedule.segments[-1].stop == 128


def test_discarded_wide_session_cannot_masquerade_as_s1_fallback(
        tmp_path: Path) -> None:
    server = _server_module()
    handler = _handler(
        server.prefill_runtime_contract, tmp_path / "s16.om")
    session, trace, _s1_positions, _head_handoffs = \
        _manual_session(server, handler)
    session._wide_session_discarded = True

    with pytest.raises(RuntimeError, match="rebuild"):
        session.generate([1, 2], 1, set(), reuse_prefix=True)

    assert trace == []

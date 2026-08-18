"""Windowed native-attention KV ingress: one test per board-measured ABI.

The three ABIs are numerically identical containers that differ 16x in host
feed cost.  A silent mismatch therefore cannot be caught by a numeric gate --
an H2 container fed H16 bytes executes and returns plausible garbage -- so
every binding is checked against the LIVE descriptor and every ABI has its own
materialisation test.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
APP_SRC = PROJECT / "app" / "src"
SOURCE = APP_SRC / "minicpm_prefill_runtime.py"

CONTEXT = 4096
LAYER = 3


def _module():
    sys.path.insert(0, str(APP_SRC))
    spec = importlib.util.spec_from_file_location(
        "pico_minicpm5_windowed_kv_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runtime():
    return _module()


def _filled_mirror(runtime, *, context=CONTEXT):
    """A packed-FP16 mirror whose every row is uniquely identifiable."""
    mirror = runtime.PackedKVCacheMirror(context)
    row = struct.Struct(f"<{runtime.HEAD_DIM}e")
    for tensor, cache in (("k", mirror.k), ("v", mirror.v)):
        for channel in range(mirror.channels):
            base = channel * mirror.channel_stride
            for index in range(mirror.past):
                seed = (channel * 7 + index * 3 + (11 if tensor == "v" else 0))
                values = [float((seed + i) % 61) - 30.0
                          for i in range(runtime.HEAD_DIM)]
                cache[base + index * runtime.ROW_F16_BYTES:
                      base + (index + 1) * runtime.ROW_F16_BYTES] = row.pack(*values)
    return mirror


def _spec(runtime, abi, *, windows=15, model_index=2, context=CONTEXT):
    binding = runtime.WindowedKVBinding(abi=abi, window_count=windows)
    return runtime.WindowedKVHandleSpec(
        width=16, context=context, model_index=model_index,
        model_path=Path("/opt/pico-minicpm5/offline/s16_attn.om"),
        kv=binding, hidden_input_bytes=131072,
        output_bytes=tuple([131072] * 15), layer_index=LAYER)


def _descriptor(runtime, abi, *, windows=15, hidden=131072):
    inputs = [hidden] + [abi.bytes_per_window] * windows + [1216, 3645440]
    return (tuple(inputs), tuple([131072] * 15))


# --------------------------------------------------------------------------
# the three ABIs are exactly the three board-measured sizes
# --------------------------------------------------------------------------

def test_the_three_known_abis_have_the_board_measured_window_sizes(runtime):
    assert runtime.KV_ABI_H16_F32.bytes_per_window == 4194304
    assert runtime.KV_ABI_H16_F16.bytes_per_window == 2097152
    assert runtime.KV_ABI_H2_F16.bytes_per_window == 262144
    assert set(runtime.KV_WINDOW_ABI_BY_BYTES) == {4194304, 2097152, 262144}
    assert runtime.KV_ABI_H2_F16.broadcast == 1
    assert runtime.KV_ABI_H16_F16.broadcast == 8
    assert runtime.KV_ABI_H16_F32.broadcast == 8


# --------------------------------------------------------------------------
# ABI 1 of 3 -- H2 FP16: the product change, a straight slice
# --------------------------------------------------------------------------

def test_h2_f16_window_is_a_straight_slice_of_the_packed_cache(runtime):
    mirror = _filled_mirror(runtime)
    handler = runtime.WindowedKVPrefillHandler(
        _spec(runtime, runtime.KV_ABI_H2_F16, windows=2))
    writes = handler.materialise(cache=mirror, start=0)

    assert tuple(write.input_slot for write in writes) == (1, 2)
    for index, write in enumerate(writes):
        assert len(write.payload) == 262144
        k_planes, v_planes = mirror.window_planes(
            layer=LAYER, start=index * runtime.KV_WINDOW_ROWS)
        # no repeat, no cast: the payload IS the four cache slices concatenated
        assert write.payload == b"".join(
            bytes(plane) for plane in (*k_planes, *v_planes))


def test_h2_f16_slices_track_the_absolute_window_start(runtime):
    mirror = _filled_mirror(runtime)
    handler = runtime.WindowedKVPrefillHandler(
        _spec(runtime, runtime.KV_ABI_H2_F16, windows=1))
    first = handler.materialise(cache=mirror, start=0)[0].payload
    later = handler.materialise(cache=mirror, start=512)[0].payload
    assert first != later
    k_planes, v_planes = mirror.window_planes(layer=LAYER, start=512)
    assert later == b"".join(bytes(p) for p in (*k_planes, *v_planes))


# --------------------------------------------------------------------------
# ABI 2 of 3 -- H16 FP16: the landed 2x, plane repeat order
# --------------------------------------------------------------------------

def test_h16_f16_window_is_the_h2_slice_fanned_out_in_repeat_order(runtime):
    mirror = _filled_mirror(runtime)
    handler = runtime.WindowedKVPrefillHandler(
        _spec(runtime, runtime.KV_ABI_H16_F16, windows=1))
    payload = handler.materialise(cache=mirror, start=0)[0].payload
    assert len(payload) == 2097152

    k_planes, v_planes = mirror.window_planes(layer=LAYER, start=0)
    plane = runtime.PLANE_F16_BYTES
    # planes 0-7 = KV head 0, 8-15 = KV head 1 (repeat order, not tile order)
    for index in range(16):
        source = k_planes[0] if index < 8 else k_planes[1]
        assert payload[index * plane:(index + 1) * plane] == bytes(source)
    for index in range(16):
        source = v_planes[0] if index < 8 else v_planes[1]
        offset = (16 + index) * plane
        assert payload[offset:offset + plane] == bytes(source)


def test_h16_f16_repeat_order_is_not_tile_order(runtime):
    """The negative control for the plane-order law."""
    mirror = _filled_mirror(runtime)
    handler = runtime.WindowedKVPrefillHandler(
        _spec(runtime, runtime.KV_ABI_H16_F16, windows=1))
    payload = handler.materialise(cache=mirror, start=0)[0].payload
    k_planes, _ = mirror.window_planes(layer=LAYER, start=0)
    plane = runtime.PLANE_F16_BYTES
    tile = b"".join(bytes(k_planes[i % 2]) for i in range(16))
    assert payload[:16 * plane] != tile


# --------------------------------------------------------------------------
# ABI 3 of 3 -- H16 FP32: the shipped form, repeat then widen
# --------------------------------------------------------------------------

def test_h16_f32_window_is_the_h16_f16_window_widened(runtime):
    mirror = _filled_mirror(runtime)
    f16 = runtime.WindowedKVPrefillHandler(
        _spec(runtime, runtime.KV_ABI_H16_F16, windows=1)
    ).materialise(cache=mirror, start=0)[0].payload
    f32 = runtime.WindowedKVPrefillHandler(
        _spec(runtime, runtime.KV_ABI_H16_F32, windows=1)
    ).materialise(cache=mirror, start=0)[0].payload

    assert len(f32) == 4194304 == 2 * len(f16)
    count = len(f16) // 2
    assert struct.unpack(f"<{count}f", f32) \
        == struct.unpack(f"<{count}e", f16)


# --------------------------------------------------------------------------
# fail-closed: a declared ABI must match the live descriptor, per window
# --------------------------------------------------------------------------

@pytest.mark.parametrize("declared_name", ["h2-f16", "h16-f16", "h16-f32"])
def test_binding_accepts_only_its_own_live_descriptor(runtime, declared_name):
    declared = runtime.KV_WINDOW_ABI_BY_NAME[declared_name]
    spec = _spec(runtime, declared)
    spec.verify_loaded_descriptor(_descriptor(runtime, declared))

    for other in runtime.KV_WINDOW_ABIS:
        if other is declared:
            continue
        with pytest.raises(runtime.WindowedKVABIError) as caught:
            spec.verify_loaded_descriptor(_descriptor(runtime, other))
        message = str(caught.value)
        assert other.describe() in message and declared.describe() in message
        assert "refusing to bind" in message


def test_unknown_window_size_is_named_as_unknown(runtime):
    spec = _spec(runtime, runtime.KV_ABI_H2_F16, windows=2)
    inputs = (131072, 262144, 524288, 1216)
    with pytest.raises(runtime.WindowedKVABIError) as caught:
        spec.verify_loaded_descriptor((inputs, tuple([131072] * 15)))
    assert "unknown ABI (524288 B/window)" in str(caught.value)
    assert "KV window 1 (input slot 2)" in str(caught.value)


def test_short_descriptor_is_refused(runtime):
    spec = _spec(runtime, runtime.KV_ABI_H2_F16, windows=15)
    inputs = (131072,) + (262144,) * 4
    with pytest.raises(runtime.WindowedKVABIError):
        spec.verify_loaded_descriptor((inputs, tuple([131072] * 15)))


def test_hidden_input_drift_is_refused(runtime):
    spec = _spec(runtime, runtime.KV_ABI_H2_F16, windows=15)
    inputs, outputs = _descriptor(runtime, runtime.KV_ABI_H2_F16, hidden=65536)
    with pytest.raises(runtime.WindowedKVABIError):
        spec.verify_loaded_descriptor((inputs, outputs))


def test_an_unregistered_abi_cannot_be_declared(runtime):
    rogue = runtime.WindowedKVABI(
        name="h4-f16", planes_per_tensor=4, element_bytes=2)
    with pytest.raises(runtime.WindowedKVABIError):
        runtime.WindowedKVBinding(abi=rogue, window_count=16)


# --------------------------------------------------------------------------
# handler / transport plumbing
# --------------------------------------------------------------------------

class _FakeTransport:
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.frames = []

    def windowed_kv_execute(self, *, spec, writes):
        spec.verify_loaded_descriptor(self.descriptor)
        self.frames.append(tuple(
            (write.input_slot, len(write.payload)) for write in writes))


@pytest.mark.parametrize("abi_name,per_window", [
    ("h2-f16", 262144), ("h16-f16", 2097152), ("h16-f32", 4194304)])
def test_execute_feeds_exactly_the_declared_bytes(runtime, abi_name, per_window):
    abi = runtime.KV_WINDOW_ABI_BY_NAME[abi_name]
    mirror = _filled_mirror(runtime)
    spec = _spec(runtime, abi, windows=2)
    transport = _FakeTransport(_descriptor(runtime, abi, windows=2))
    result = runtime.WindowedKVPrefillHandler(spec).execute(
        transport=transport, cache=mirror, start=0, hidden=b"\x00" * 131072)

    assert result.abi == abi_name
    assert result.feed_bytes == 131072 + 2 * per_window
    assert transport.frames == [
        ((0, 131072), (1, per_window), (2, per_window))]
    assert result.materialise_ms >= 0.0 and result.execute_ms >= 0.0


def test_execute_refuses_a_mismatched_live_descriptor(runtime):
    mirror = _filled_mirror(runtime)
    spec = _spec(runtime, runtime.KV_ABI_H2_F16, windows=2)
    # the container on the table is the shipped H16 f32 one
    transport = _FakeTransport(
        _descriptor(runtime, runtime.KV_ABI_H16_F32, windows=2))
    with pytest.raises(runtime.WindowedKVABIError):
        runtime.WindowedKVPrefillHandler(spec).execute(
            transport=transport, cache=mirror, start=0)
    assert transport.frames == []


def test_window_slicing_cannot_escape_the_context_minus_one_cache(runtime):
    mirror = _filled_mirror(runtime)
    handler = runtime.WindowedKVPrefillHandler(
        _spec(runtime, runtime.KV_ABI_H2_F16, windows=15))
    handler.materialise(cache=mirror, start=0)          # rows [0, 3840)
    with pytest.raises(runtime.WindowedKVABIError,
                       match="escapes the context-1 resident cache"):
        handler.materialise(cache=mirror, start=512)    # would need row 4351


def test_a_sixteen_window_rung_needs_a_context_of_at_least_4097(runtime):
    """The shipped ctx-4096 canonical cache is exactly one row short."""
    binding = runtime.WindowedKVBinding(
        abi=runtime.KV_ABI_H2_F16, window_count=16)
    assert binding.kv_rows == 4096
    with pytest.raises(runtime.WindowedKVABIError, match="at least 4097"):
        _spec(runtime, runtime.KV_ABI_H2_F16, windows=16, context=4096)
    wide = _spec(runtime, runtime.KV_ABI_H2_F16, windows=16, context=16384)
    assert wide.public_input_count == 17
    assert wide.kv.bytes_per_layer == 4194304


def test_mirror_context_must_match_the_handle(runtime):
    mirror = _filled_mirror(runtime, context=1024)
    handler = runtime.WindowedKVPrefillHandler(
        _spec(runtime, runtime.KV_ABI_H2_F16, windows=1))
    with pytest.raises(runtime.WindowedKVABIError):
        handler.materialise(cache=mirror, start=0)


def test_feed_rejects_a_wrong_sized_head_plane(runtime):
    feed = runtime.WindowKVFeed(runtime.KV_ABI_H2_F16)
    good = b"\x00" * runtime.PLANE_F16_BYTES
    with pytest.raises(runtime.WindowedKVABIError):
        feed.payload((good, good[:-2]), (good, good))
    with pytest.raises(runtime.WindowedKVABIError):
        feed.payload((good,), (good, good))


# --------------------------------------------------------------------------
# server transport: the frame is only written after the live descriptor agrees
# --------------------------------------------------------------------------

def _server_module():
    sys.path.insert(0, str(APP_SRC))
    spec = importlib.util.spec_from_file_location(
        "pico_minicpm5_windowed_kv_server_test",
        APP_SRC / "merged_board_server.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _session(server, runtime, abi, *, windows=16):
    import io
    from types import SimpleNamespace

    session = object.__new__(server.Merged)
    session.models = [Path("decode.om"), Path("s16_attn.om")]
    inputs, outputs = _descriptor(runtime, abi, windows=windows)
    session.descriptors = [((1,) * 7, (1, 1, 1)), (inputs, outputs)]
    session.process = SimpleNamespace(stdin=io.BytesIO())
    session.process.stdin.flush = lambda: None
    session._wide_session_discarded = False
    session._respond = lambda sizes, model=None: []
    return session


def test_server_frame_declares_the_containers_own_public_input_count():
    server = _server_module()
    runtime = server.prefill_runtime_contract
    abi = runtime.KV_ABI_H2_F16
    session = _session(server, runtime, abi, windows=16)
    # 4097 is the exact minimum a 16-window rung can be hosted at
    spec = _spec(runtime, abi, windows=16, model_index=1, context=4097)
    mirror = runtime.PackedKVCacheMirror(4097)
    writes = runtime.WindowedKVPrefillHandler(spec).materialise(
        cache=mirror, start=0, hidden=b"\x00" * 131072)

    session.windowed_kv_execute(spec=spec, writes=writes)

    frame = session.process.stdin.getvalue()
    header = server.runner._PERSISTENT_REQUEST
    (magic, _version, opcode, model, public_inputs, out_count,
     write_count) = header.unpack(frame[:header.size])
    assert magic == server.runner.PERSISTENT_REQUEST_MAGIC
    assert opcode == server.runner.PERSISTENT_OP_EXECUTE_RESIDENT
    assert (model, public_inputs, out_count, write_count) == (1, 17, 0, 17)
    # 17 writes: hidden + 16 windows, and not one byte more than the H2 ABI
    assert len(frame) == header.size \
        + 17 * server.runner._PERSISTENT_WRITE.size \
        + 131072 + 16 * 262144


@pytest.mark.parametrize("live_name", ["h16-f16", "h16-f32"])
def test_server_refuses_to_bind_a_container_of_another_abi(live_name):
    server = _server_module()
    runtime = server.prefill_runtime_contract
    declared = runtime.KV_ABI_H2_F16
    live = runtime.KV_WINDOW_ABI_BY_NAME[live_name]
    session = _session(server, runtime, live, windows=4)
    spec = _spec(runtime, declared, windows=4, model_index=1)
    mirror = runtime.PackedKVCacheMirror(CONTEXT)
    writes = runtime.WindowedKVPrefillHandler(spec).materialise(
        cache=mirror, start=0, hidden=b"\x00" * 131072)

    with pytest.raises(runtime.WindowedKVABIError, match="refusing to bind"):
        session.windowed_kv_execute(spec=spec, writes=writes)
    assert session.process.stdin.getvalue() == b""


def test_server_refuses_a_payload_that_is_not_the_declared_window_size():
    server = _server_module()
    runtime = server.prefill_runtime_contract
    abi = runtime.KV_ABI_H2_F16
    session = _session(server, runtime, abi, windows=2)
    spec = _spec(runtime, abi, windows=2, model_index=1)
    writes = (
        runtime.WideInputWrite(0, 0, b"\x00" * 131072),
        runtime.WideInputWrite(1, 0, b"\x00" * 262144),
        runtime.WideInputWrite(2, 0, b"\x00" * 2097152),   # H16 bytes
    )
    with pytest.raises(RuntimeError, match="262144"):
        session.windowed_kv_execute(spec=spec, writes=writes)
    assert session.process.stdin.getvalue() == b""


def test_server_refuses_an_incomplete_window_set():
    server = _server_module()
    runtime = server.prefill_runtime_contract
    abi = runtime.KV_ABI_H2_F16
    session = _session(server, runtime, abi, windows=2)
    spec = _spec(runtime, abi, windows=2, model_index=1)
    writes = (
        runtime.WideInputWrite(0, 0, b"\x00" * 131072),
        runtime.WideInputWrite(1, 0, b"\x00" * 262144),
    )
    with pytest.raises(RuntimeError, match="every declared window slot"):
        session.windowed_kv_execute(spec=spec, writes=writes)
    assert session.process.stdin.getvalue() == b""


def test_canonical_five_input_handles_keep_their_public_input_count():
    """The public-input parameter is additive: the decode default is unchanged."""
    server = _server_module()
    runtime = server.prefill_runtime_contract
    session = _session(server, runtime, runtime.KV_ABI_H2_F16, windows=2)
    session._run(0, ((0, 0, b"\x01\x02"),), publish=False, output_count=0)
    header = server.runner._PERSISTENT_REQUEST
    frame = session.process.stdin.getvalue()
    assert header.unpack(frame[:header.size])[4] == 5

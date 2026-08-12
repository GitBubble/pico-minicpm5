from __future__ import annotations

import io
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PROJECT = Path(__file__).resolve().parents[1]
APP_SRC = PROJECT / "app" / "src"
sys.path.insert(0, str(APP_SRC))

import pico_minicpm5_split_board_runner as runner  # noqa: E402


def _executor(*, input_bytes: int) -> runner.PersistentAclExecutor:
    executor = object.__new__(runner.PersistentAclExecutor)
    executor._closed = False
    executor.process = SimpleNamespace(stdin=io.BytesIO())
    inputs = (64, 64, 64, input_bytes, input_bytes)
    executor.descriptors = ((inputs, (64,)), (inputs, (64,)))
    executor.model_paths = (Path("/model0.om"), Path("/model1.om"))
    return executor


def test_resident_input_copy_encodes_96_channel_records_atomically() -> None:
    past = 1023
    row_bytes = 256
    cache_bytes = 48 * past * row_bytes
    start = 643
    executor = _executor(input_bytes=cache_bytes)
    responses: list[tuple[int, tuple[int, ...]]] = []
    executor._read_response = lambda model, sizes: (
        responses.append((model, tuple(sizes))) or ())

    records = []
    for input_index in (3, 4):
        for channel in range(48):
            offset = (channel * past + start) * row_bytes
            records.append(runner.ResidentInputCopy(
                destination_model=0,
                destination_input=input_index,
                destination_offset=offset,
                source_model=1,
                source_input=input_index,
                source_offset=offset,
                length=row_bytes,
            ))
    executor.copy_resident_inputs(records, tag="prefill-cache-fanout")

    payload = executor.process.stdin.getvalue()
    header_size = runner._PERSISTENT_REQUEST.size
    record_size = runner._PERSISTENT_INPUT_COPY.size
    assert record_size == 44
    assert len(payload) == header_size + 96 * record_size
    assert runner._PERSISTENT_REQUEST.unpack(payload[:header_size]) == (
        runner.PERSISTENT_REQUEST_MAGIC,
        runner.PERSISTENT_PROTOCOL_VERSION,
        runner.PERSISTENT_OP_COPY_INPUTS,
        0,
        96,
        0,
        0,
    )
    first = runner._PERSISTENT_INPUT_COPY.unpack(
        payload[header_size:header_size + record_size])
    last = runner._PERSISTENT_INPUT_COPY.unpack(payload[-record_size:])
    assert first == (0, 3, start * row_bytes,
                     1, 3, start * row_bytes, row_bytes, 0)
    last_offset = (47 * past + start) * row_bytes
    assert last == (0, 4, last_offset,
                    1, 4, last_offset, row_bytes, 0)
    assert responses == [(0, ())]


def test_resident_input_copy_bad_late_record_emits_no_partial_frame() -> None:
    executor = _executor(input_bytes=1024)
    executor._read_response = lambda *_args: pytest.fail(
        "invalid records must not reach the response phase")
    records = (
        runner.ResidentInputCopy(0, 3, 0, 1, 3, 0, 128),
        # The second record fails exact destination bounds.  Validation of the
        # complete list happens before the request header or first record is
        # written, so the protocol pipe remains untouched.
        runner.ResidentInputCopy(0, 4, 1023, 1, 4, 0, 2),
    )

    with pytest.raises(
            runner.ExecutionError,
            match=r"record\[1\] exceeds source or destination input"):
        executor.copy_resident_inputs(records)
    assert executor.process.stdin.getvalue() == b""


@pytest.mark.parametrize(
    "record, message",
    [
        (runner.ResidentInputCopy(2, 0, 0, 1, 0, 0, 1),
         "model index is out of range"),
        (runner.ResidentInputCopy(0, 5, 0, 1, 0, 0, 1),
         "input index is out of range"),
        (runner.ResidentInputCopy(0, 0, 0, 1, 0, 0, 0),
         "offset or length is invalid"),
    ],
)
def test_resident_input_copy_rejects_invalid_contract_before_wire(
    record: runner.ResidentInputCopy, message: str,
) -> None:
    executor = _executor(input_bytes=1024)
    with pytest.raises(runner.ExecutionError, match=message):
        executor.copy_resident_inputs((record,))
    assert executor.process.stdin.getvalue() == b""


def test_resident_input_copy_keeps_existing_opcode_numbers_stable() -> None:
    assert (
        runner.PERSISTENT_OP_EXECUTE,
        runner.PERSISTENT_OP_SHUTDOWN,
        runner.PERSISTENT_OP_WRITE_INPUT,
        runner.PERSISTENT_OP_EXECUTE_RESIDENT,
        runner.PERSISTENT_OP_ARGMAX,
        runner.PERSISTENT_OP_SCATTER_F32_TO_F16,
        runner.PERSISTENT_OP_SNAPSHOT_INPUTS,
        runner.PERSISTENT_OP_RESTORE_INPUTS,
        runner.PERSISTENT_OP_COPY_INPUTS,
    ) == tuple(range(1, 10))

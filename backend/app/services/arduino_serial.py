"""Arduino/AD8232 serial discovery and raw acquisition helpers.

This module intentionally performs no ECG filtering or ML preprocessing. It is
the hardware-validation layer used before connecting serial data to live model
inference.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import numpy as np


@dataclass(frozen=True)
class ArduinoSample:
    ecg: int
    lead_off: Optional[bool]


_NAMED_ECG = re.compile(r"(?:^|\s)ECG\s*:\s*(-?\d+)(?:\s|$)", re.IGNORECASE)
_NAMED_LEAD = re.compile(r"(?:^|\s)LeadOff\s*:\s*(-?\d+)(?:\s|$)", re.IGNORECASE)


def parse_sample_line(line: str) -> Optional[ArduinoSample]:
    """Parse the standalone plotter sketch and a few simple future formats.

    Accepted examples:
      ECG:512\tLeadOff:0\tMin:0\tMax:1023
      512
      1234,512,0   (sample_number, ADC value, lead-off flag)
    """
    text = line.strip()
    if not text:
        return None

    named_ecg = _NAMED_ECG.search(text)
    if named_ecg:
        ecg = int(named_ecg.group(1))
        named_lead = _NAMED_LEAD.search(text)
        lead_off = bool(int(named_lead.group(1))) if named_lead else None
        return ArduinoSample(ecg=ecg, lead_off=lead_off)

    if re.fullmatch(r"-?\d+", text):
        return ArduinoSample(ecg=int(text), lead_off=None)

    fields = [field.strip() for field in text.split(",")]
    if len(fields) >= 3 and all(re.fullmatch(r"-?\d+", field) for field in fields[:3]):
        return ArduinoSample(ecg=int(fields[1]), lead_off=bool(int(fields[2])))
    return None


def list_serial_ports() -> List[Dict[str, Any]]:
    """Return serial ports reported by the operating system."""
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError("pyserial is not installed; run pip install -r requirements.txt") from exc

    ports = []
    for port in sorted(list_ports.comports(), key=lambda item: item.device):
        ports.append({
            "device": port.device,
            "description": port.description or "",
            "manufacturer": port.manufacturer or "",
            "hwid": port.hwid or "",
            "vid": port.vid,
            "pid": port.pid,
        })
    return ports


def acquire_test_samples(
    port: str,
    baud_rate: int,
    sample_count: int,
    timeout_seconds: float,
) -> Dict[str, Any]:
    """Open a port exclusively and collect unprocessed ADC samples."""
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is not installed; run pip install -r requirements.txt") from exc

    samples: List[int] = []
    lead_off_samples = 0
    samples_with_lead_status = 0
    malformed_lines = 0
    first_sample_at: Optional[float] = None
    last_sample_at: Optional[float] = None
    started_at = time.monotonic()
    deadline = started_at + timeout_seconds

    # A short readline timeout lets us enforce the request's overall deadline.
    with serial.Serial(port=port, baudrate=baud_rate, timeout=0.25) as connection:
        # Opening a classic Nano normally toggles DTR and resets it. Discard any
        # partial boot-time line, while the overall deadline remains active.
        connection.reset_input_buffer()

        while len(samples) < sample_count and time.monotonic() < deadline:
            raw = connection.readline()
            if not raw:
                continue
            try:
                line = raw.decode("ascii", errors="strict")
            except UnicodeDecodeError:
                malformed_lines += 1
                continue

            sample = parse_sample_line(line)
            if sample is None:
                malformed_lines += 1
                continue

            received_at = time.monotonic()
            if first_sample_at is None:
                first_sample_at = received_at
            last_sample_at = received_at
            samples.append(sample.ecg)
            if sample.lead_off is not None:
                samples_with_lead_status += 1
                lead_off_samples += int(sample.lead_off)

    if not samples:
        raise RuntimeError(
            f"No valid ECG samples received from {port}. Check the port, 115200 baud, "
            "uploaded sketch, and ensure Serial Plotter/Monitor is closed."
        )

    elapsed = max(time.monotonic() - started_at, 1e-9)
    acquisition_span = (
        last_sample_at - first_sample_at
        if first_sample_at is not None and last_sample_at is not None
        else 0.0
    )
    estimated_hz = (
        (len(samples) - 1) / acquisition_span
        if len(samples) > 1 and acquisition_span > 0
        else None
    )
    minimum = min(samples)
    maximum = max(samples)

    return {
        "port": port,
        "baud_rate": baud_rate,
        "requested_samples": sample_count,
        "received_samples": len(samples),
        "timed_out": len(samples) < sample_count,
        "elapsed_seconds": round(elapsed, 3),
        "estimated_sample_rate_hz": round(estimated_hz, 2) if estimated_hz else None,
        "malformed_lines": malformed_lines,
        "lead_off_samples": lead_off_samples,
        "samples_with_lead_status": samples_with_lead_status,
        "lead_off_detected": lead_off_samples > 0,
        "minimum": minimum,
        "maximum": maximum,
        "mean": round(sum(samples) / len(samples), 3),
        "peak_to_peak": maximum - minimum,
        "clipped_low_samples": sum(value <= 2 for value in samples),
        "clipped_high_samples": sum(value >= 1021 for value in samples),
        "samples": samples,
    }


class ArduinoSerialStream(Iterator[np.ndarray]):
    """Exclusive, blocking Arduino stream that yields raw ADC chunks."""

    def __init__(self, port: str, baud_rate: int = 115200,
                 sample_rate_hz: int = 128, chunk_seconds: float = 4.0) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is not installed; run pip install -r requirements.txt") from exc
        self.port = port
        self.sample_rate_hz = sample_rate_hz
        self.chunk_samples = int(round(sample_rate_hz * chunk_seconds))
        self.connection = serial.Serial(port=port, baudrate=baud_rate, timeout=0.25)
        self.connection.reset_input_buffer()
        self.closed = False
        self.last_status: Dict[str, Any] = {}

    def __iter__(self) -> "ArduinoSerialStream":
        return self

    def __next__(self) -> np.ndarray:
        if self.closed or not self.connection.is_open:
            raise StopIteration
        values: List[int] = []
        malformed = lead_off_count = lead_status_count = 0
        expected_seconds = self.chunk_samples / self.sample_rate_hz
        deadline = time.monotonic() + max(5.0, expected_seconds * 2.0 + 2.0)
        while len(values) < self.chunk_samples and time.monotonic() < deadline:
            raw = self.connection.readline()
            if not raw:
                continue
            try:
                sample = parse_sample_line(raw.decode("ascii", errors="strict"))
            except UnicodeDecodeError:
                sample = None
            if sample is None:
                malformed += 1
                continue
            values.append(sample.ecg)
            if sample.lead_off is not None:
                lead_status_count += 1
                lead_off_count += int(sample.lead_off)
        if len(values) != self.chunk_samples:
            raise RuntimeError(
                f"Serial timeout on {self.port}: received {len(values)}/{self.chunk_samples} samples. "
                "Check the USB connection and close other serial applications."
            )
        self.last_status = {
            "port": self.port,
            "lead_off": lead_off_count > 0,
            "lead_off_samples": lead_off_count,
            "samples_with_lead_status": lead_status_count,
            "malformed_lines": malformed,
            "minimum": min(values), "maximum": max(values),
            "clipped_low_samples": sum(value <= 2 for value in values),
            "clipped_high_samples": sum(value >= 1021 for value in values),
        }
        return np.asarray(values, dtype=np.float32)

    def close(self) -> None:
        self.closed = True
        if self.connection.is_open:
            self.connection.close()

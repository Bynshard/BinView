#!/usr/bin/env python3
import json
import math
import struct
import sys
from dataclasses import dataclass
from typing import Any


try:
    import torch
except Exception as exc:  # pragma: no cover - runtime dependency
    sys.stderr.write(f"Failed to import torch: {exc}\n")
    sys.stderr.flush()
    raise


@dataclass
class Candidate:
    candidate_id: int
    label: str
    tensor: Any


class Session:
    def __init__(self) -> None:
        self._root = None
        self._candidates: list[Candidate] = []
        self._base = None
        self._base_label = "identity"
        self._current = None
        self._notes: list[str] = []
        self._transform_summary = "identity"

    def open(self, path: str) -> dict[str, Any]:
        self._notes = []
        self._transform_summary = "identity"
        try:
            self._root = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            self._root = torch.load(path, map_location="cpu")

        self._candidates = []
        self._collect_candidates(self._root, "root", 0)
        return {
            "candidates": [
                {
                    "id": candidate.candidate_id,
                    "label": candidate.label,
                    "dtype": str(candidate.tensor.dtype),
                    "shape": list(candidate.tensor.shape),
                }
                for candidate in self._candidates
            ],
            "dtype": "",
            "shape": [],
            "totalElements": 0,
            "notes": self._notes,
        }

    def select(self, candidate_id: int) -> dict[str, Any]:
        for candidate in self._candidates:
            if candidate.candidate_id == candidate_id:
                self._base = candidate.tensor.detach().cpu()
                self._base_label = candidate.label
                self._current = self._base
                self._transform_summary = candidate.label
                return self._describe()
        raise ValueError(f"Unknown tensor candidate: {candidate_id}")

    def reshape(self, shape: list[int]) -> dict[str, Any]:
        if self._current is None:
            raise RuntimeError("Tensor has not been selected.")
        self._current = self._current.reshape(shape)
        self._transform_summary += f" -> reshape({shape})"
        return self._describe()

    def slice(self, expression: str) -> dict[str, Any]:
        if self._current is None:
            raise RuntimeError("Tensor has not been selected.")
        self._current = eval(f"self._current[{expression}]", {"__builtins__": {}}, {"self": self})
        self._transform_summary += f" -> [{expression}]"
        return self._describe()

    def reset(self) -> dict[str, Any]:
        if self._base is None:
            raise RuntimeError("Tensor has not been selected.")
        self._current = self._base
        self._transform_summary = self._base_label
        return self._describe()

    def fetch(self, start: int, count: int) -> list[dict[str, Any]]:
        if self._current is None:
            raise RuntimeError("Tensor has not been selected.")

        flat = self._current.reshape(-1)
        start = max(0, min(start, int(flat.numel())))
        end = max(start, min(start + count, int(flat.numel())))
        chunk = flat[start:end].detach().cpu().contiguous()

        return [
            {
                "index": start + idx,
                "value": self._format_scalar(value),
                "hex": self._format_hex(value),
                "bits": self._format_bits(value),
            }
            for idx, value in enumerate(chunk)
        ]

    def close(self) -> dict[str, Any]:
        return {}

    def _describe(self) -> dict[str, Any]:
        if self._current is None:
            raise RuntimeError("Tensor has not been selected.")
        return {
            "dtype": str(self._current.dtype),
            "shape": list(self._current.shape),
            "totalElements": int(self._current.numel()),
            "transformSummary": self._transform_summary,
            "notes": self._notes,
        }

    def _collect_candidates(self, obj: Any, label: str, depth: int) -> None:
        if depth > 6 or len(self._candidates) >= 128:
            return
        if torch.is_tensor(obj):
            self._candidates.append(Candidate(len(self._candidates), label, obj))
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                self._collect_candidates(value, f"{label}[{key!r}]", depth + 1)
            return
        if isinstance(obj, (list, tuple)):
            for index, value in enumerate(obj):
                self._collect_candidates(value, f"{label}[{index}]", depth + 1)

    def _format_scalar(self, value: Any) -> str:
        item = value.item()
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, float):
            if math.isnan(item):
                return "NaN"
            if math.isinf(item):
                return "Infinity" if item > 0 else "-Infinity"
            if item == 0.0 and math.copysign(1.0, item) < 0:
                return "-0"
            absolute = abs(item)
            if absolute >= 1_000_000 or (absolute > 0 and absolute < 0.0001):
                return f"{item:.7e}"
            if item.is_integer():
                return str(int(item))
            return f"{item:.9g}"
        return str(item)

    def _pack_scalar(self, value: Any) -> bytes:
        dtype = str(value.dtype)
        item = value.item()
        if dtype == "torch.bool":
            return bytes([1 if item else 0])
        if dtype == "torch.int8":
            return struct.pack("<b", item)
        if dtype == "torch.uint8":
            return struct.pack("<B", item)
        if dtype == "torch.int16":
            return struct.pack("<h", item)
        if dtype == "torch.int32":
            return struct.pack("<i", item)
        if dtype == "torch.int64":
            return struct.pack("<q", item)
        if dtype == "torch.float16":
            return struct.pack("<e", item)
        if dtype == "torch.float32":
            return struct.pack("<f", item)
        if dtype == "torch.float64":
            return struct.pack("<d", item)
        if dtype == "torch.bfloat16":
            float_bytes = struct.pack("<f", float(item))
            return float_bytes[2:]
        raise ValueError(f"Unsupported torch dtype: {dtype}")

    def _format_hex(self, value: Any) -> str:
        packed = self._pack_scalar(value)
        return " ".join(f"{byte:02x}" for byte in packed)

    def _format_bits(self, value: Any) -> str:
        packed = self._pack_scalar(value)
        return " ".join(f"{byte:08b}" for byte in packed)


def main() -> int:
    session = Session()
    handlers = {
        "open": lambda payload: session.open(payload["path"]),
        "select": lambda payload: session.select(payload["candidateId"]),
        "reshape": lambda payload: session.reshape(payload["shape"]),
        "slice": lambda payload: session.slice(payload["expression"]),
        "reset": lambda payload: session.reset(),
        "fetch": lambda payload: session.fetch(payload["start"], payload["count"]),
        "close": lambda payload: session.close(),
    }

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        request = json.loads(line)
        request_id = request["id"]
        command = request["command"]
        payload = {key: value for key, value in request.items() if key not in {"id", "command"}}

        try:
            if command not in handlers:
                raise ValueError(f"Unknown command: {command}")
            result = handlers[command](payload)
            response = {"id": request_id, "ok": True, "result": result}
        except Exception as exc:
            response = {"id": request_id, "ok": False, "error": str(exc)}

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Per-frame CSV recording of signals and scores, for offline evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import IO, Any

from drowsiness.pipeline import FrameReport


class SignalRecorder:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] = self.path.open("w", newline="", encoding="utf-8")
        self._writer: csv.DictWriter[str] | None = None
        self.rows = 0

    def write(self, report: FrameReport, **extra: Any) -> None:
        row = {**report.to_row(), **extra}
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self.rows += 1

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> SignalRecorder:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

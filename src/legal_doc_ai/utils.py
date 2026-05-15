
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            if hasattr(record, "model_dump"):
                record = record.model_dump()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def stable_doc_id(path: str | Path) -> str:
    path = Path(path)
    value = path.stem.lower().replace(" ", "_").replace("-", "_")
    return "".join(ch for ch in value if ch.isalnum() or ch == "_")

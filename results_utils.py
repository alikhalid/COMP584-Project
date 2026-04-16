import json
from dataclasses import asdict, is_dataclass
from pathlib import Path


def _to_serializable(value):
    if is_dataclass(value):
        return _to_serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def save_json(path, payload):
    path = Path(path)
    path.write_text(json.dumps(_to_serializable(payload), indent=2), encoding="utf-8")


def save_text(path, text):
    path = Path(path)
    path.write_text(text, encoding="utf-8")


def save_experiment_artifacts(experiment_name, *, metrics, config, sample_text=None, extra=None):
    save_json(f"{experiment_name}_metrics.json", metrics)
    save_json(f"{experiment_name}_config.json", config)

    if sample_text is not None:
        save_text(f"{experiment_name}_sample.txt", sample_text)

    if extra is not None:
        save_json(f"{experiment_name}_extra.json", extra)

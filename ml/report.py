"""Write a backtest result to disk: ``metrics.json`` and a readable ``report.md``.

Reports live under ``<DATA_DIR>/reports/<task>/<UTC timestamp>/``; the newest folder is the one
the API and the dashboard show.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.config import data_dir

BLOCK_TITLE = {"walk_forward": "Walk-forward (pooled test folds)", "holdout": "Final hold-out"}


def reports_dir() -> Path:
    return data_dir() / "reports"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def render(result: dict[str, Any]) -> str:
    """Markdown report. The first lines state sizes, what persistence means, and the verdict."""
    unit = result["target"]["unit"]
    lines = [f"# Backtest report: {result['task']}", ""]
    if result["simulated"]:
        lines += [
            "> **SIMULATED DATA - pipeline validation only, not real-system performance.**",
            "",
        ]
    if result["description"]:
        lines += [result["description"].strip(), ""]
    lines += [
        f"- Target: `{result['target']['channel']}` ({unit}), forecast "
        f"{result['horizon_steps']} step(s) of {result['resolution']} ahead "
        f"({result['horizon_h']:g} h).",
        f"- Rows with a known outcome: {result['n_rows']} from nodes "
        f"{', '.join(result['nodes']) or '-'}; sources: {', '.join(result['sources']) or '-'}.",
    ]
    if result.get("data_start"):
        lines.append(f"- Data from {result['data_start']} to {result['data_end']} (UTC).")
    if result["event"]:
        e = result["event"]
        lines.append(
            f"- Event: value at or {e['direction']} {e['threshold']:g} {unit} at the valid time."
        )
    for name, meaning in result["baseline_meaning"].items():
        lines.append(f"- Baseline `{name}`: {meaning}.")
    lines.append("")

    lines += ["## Verdict", ""]
    block_name = "holdout" if "holdout" in result["blocks"] else "walk_forward"
    block = result["blocks"].get(block_name)
    if block is None:
        lines += ["Nothing could be scored. See the warnings below.", ""]
    else:
        lines.append(
            f"On the {BLOCK_TITLE[block_name].lower()} ({block['n']} rows, {block['n_days']} days):"
        )
        lines.append("")
        for entry in block["predictors"].values():
            if entry["is_baseline"]:
                continue
            for res in entry["vs"].values():
                lines.append(f"- {res['sentence']}")
        lines.append("")

    if result["warnings"] or result["skipped"]:
        lines += ["## Warnings", ""]
        lines += [f"- {w}" for w in result["warnings"]]
        lines += [f"- Skipped: {s}" for s in result["skipped"]]
        lines.append("")

    for name, block in result["blocks"].items():
        lines += [f"## {BLOCK_TITLE[name]}", ""]
        lines.append(
            f"{block['n']} test rows over {block['n_days']} days, valid "
            f"{block['first_valid_at']} to {block['last_valid_at']}."
        )
        lines += [
            "",
            f"| predictor | n | MAE ({unit}) | RMSE | bias | R2 | skill vs persistence "
            "| interval coverage |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for pname, e in block["predictors"].items():
            label = f"{pname} (baseline)" if e["is_baseline"] else pname
            lines.append(
                f"| {label} | {e['n']} | {_fmt(e['mae'])} | {_fmt(e['rmse'])} | {_fmt(e['bias'])} "
                f"| {_fmt(e['r2'])} | {_fmt(e['skill_vs_persistence'], 3)} "
                f"| {_fmt(e.get('interval_coverage'), 3)} |"
            )
        lines.append("")
        if result["event"]:
            lines += [
                f"Event rows: {block.get('event_positives', 0)} of {block['n']}. A row is flagged "
                "when the event probability is at least 0.5.",
                "",
                "| predictor | precision | recall | F1 | Brier | TP | FP | FN | TN |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for pname, e in block["predictors"].items():
                ev = e["event"]
                lines.append(
                    f"| {pname} | {_fmt(ev['precision'], 3)} | {_fmt(ev['recall'], 3)} "
                    f"| {_fmt(ev['f1'], 3)} | {_fmt(ev['brier'], 3)} | {_fmt(ev['tp'])} "
                    f"| {_fmt(ev['fp'])} | {_fmt(ev['fn'])} | {_fmt(ev['tn'])} |"
                )
            lines.append("")

    if result["importance"]:
        imp = result["importance"]
        lines += [f"## What the best walk-forward model ({imp['model']}) relies on", "",
                  f"Permutation importance on the hold-out: {imp['meaning']}.", "",
                  "| feature | MAE increase |", "|---|---|"]  # fmt: skip
        for feature, value in list(imp["features"].items())[:15]:
            lines.append(f"| `{feature}` | {_fmt(value)} |")
        lines.append("")

    lines += [
        "## Splits",
        "",
        "| fold | train rows | test rows | last train label | test issue times |",
        "|---|---|---|---|---|",
    ]
    for f in result["folds"]:
        lines.append(
            f"| {f['fold']} | {f['n_train']} | {f['n_test']} | {f['train_last_valid_at']} "
            f"| {f['test_first_issue']} to {f['test_last_issue']} |"
        )
    lines += [
        "",
        "Splits are chronological. Every training label was already known at the first test issue "
        "time. Intervals are split-conformal from walk-forward residuals "
        f"({100 * result['interval_level']:.0f}% nominal).",
        "",
    ]
    return "\n".join(lines)


def save(result: dict[str, Any]) -> Path:
    """Write the report folder and return it."""
    stamp = result["created_at"].replace(":", "").replace("-", "")[:15]
    folder = reports_dir() / result["task"] / stamp
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    (folder / "report.md").write_text(render(result), encoding="utf-8")
    return folder


def report_tasks() -> list[str]:
    root = reports_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and latest_folder(p.name))


def latest_folder(task: str) -> Path | None:
    root = reports_dir() / task
    if not root.is_dir():
        return None
    folders = sorted(p for p in root.iterdir() if (p / "metrics.json").is_file())
    return folders[-1] if folders else None


def latest(task: str) -> dict[str, Any] | None:
    """The newest stored backtest result of a task, with its folder and Markdown."""
    folder = latest_folder(task)
    if folder is None:
        return None
    result = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    result["report_path"] = str(folder)
    return result

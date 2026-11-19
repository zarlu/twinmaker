"""Command line for the ML jobs.

    python -m ml.run backtest --task <name>
    python -m ml.run train    --task <name> [--algorithm ridge] [--holdback-days 3]
    python -m ml.run promote  --version <model version>
    python -m ml.run forecast --task <name> [--backfill-steps 72]
    python -m ml.run scorecard --task <name>
    python -m ml.run tasks

Every command reads the database behind ``DB_URL`` and the project behind ``PROJECT_DIR``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from db.engine import init_db, make_engine, make_session_factory
from ml import jobs
from ml.config import TaskError, list_tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ml.run", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("backtest", "train", "forecast", "scorecard"):
        p = sub.add_parser(name)
        p.add_argument("--task", required=True)
        if name == "train":
            p.add_argument("--algorithm", default=None)
            p.add_argument("--holdback-days", type=float, default=0.0,
                           help="leave the newest N days out of training")  # fmt: skip
        if name == "forecast":
            p.add_argument("--backfill-steps", type=int, default=0,
                           help="also issue forecasts for the previous N grid times")  # fmt: skip
    sub.add_parser("promote").add_argument("--version", required=True)
    sub.add_parser("tasks")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "tasks":
        print("\n".join(list_tasks()) or "(no tasks in this project)")
        return 0
    engine = make_engine()
    init_db(engine)
    try:
        with make_session_factory(engine)() as session:
            if args.command == "backtest":
                out = jobs.backtest(session, args.task)
            elif args.command == "train":
                out = jobs.train(session, args.task, args.algorithm, args.holdback_days)
            elif args.command == "forecast":
                out = jobs.forecast(session, args.task, args.backfill_steps)
            elif args.command == "scorecard":
                out = jobs.scorecard(session, args.task)
            else:
                out = jobs.promote(session, args.version)
    except (jobs.JobError, TaskError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

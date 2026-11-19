"""Run the simulator: ``python -m sim.run --days 30`` (or ``make sim ARGS="--days 30"``).

Backfill mode (default) posts ``--days`` of history ending now and exits. With ``--live`` it then
keeps posting one time step at a time as the clock reaches it, like real nodes would.
``--dry-run`` writes the would-be requests to ``sim_out/`` instead of posting.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sim.client import DryRunWriter, SimClient
from sim.generate import generate, load_sim_config, to_messages

log = logging.getLogger("sim")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m sim.run", description=__doc__.split("\n")[0])
    p.add_argument("--days", type=float, default=30.0, help="days of history ending now")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--live", action="store_true", help="keep posting in real time after backfill")
    p.add_argument("--live-hours", type=float, default=24.0, help="how long --live runs")
    p.add_argument("--no-faults", action="store_true", help="do not inject sensor faults")
    p.add_argument("--dry-run", action="store_true", help="write JSONL to sim_out/, post nothing")
    p.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8000"))
    p.add_argument("--api-key", default=os.environ.get("API_KEY", ""))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_sim_config()
    now = datetime.now(UTC)
    start = now - timedelta(days=args.days)
    end = now + timedelta(hours=args.live_hours) if args.live else now
    frames, fault_log = generate(cfg, start, end, seed=args.seed, faults=not args.no_faults)
    if args.dry_run:
        client = DryRunWriter(Path("sim_out") / f"seed{args.seed}.jsonl")
    else:
        if not args.api_key:
            print(
                "error: API_KEY is not set (see .env) and --api-key was not given", file=sys.stderr
            )
            return 1
        client = SimClient(args.api_url, args.api_key)
    batch_rows = max(1, int(round(6 * 60 / cfg["step_minutes"])))
    past = {node: f[f.index <= now] for node, f in frames.items()}
    for message in to_messages(past, batch_rows, sent_at=now):
        client.post("/ingest", message)
    stats = client.stats
    print(
        f"backfill: {stats.accepted} accepted, {stats.duplicates} duplicates, "
        f"{stats.rejected} rejected, {stats.failed_requests} failed requests "
        f"({len(fault_log)} faults injected; all rows source='simulator')"
    )
    if args.dry_run:
        client.close()
        print(f"dry run written to {client.path}")
        return 0
    if stats.failed_requests:
        return 1
    if args.live:
        future = sorted({ts for f in frames.values() for ts in f.index[f.index > now]})
        print(f"live: posting {len(future)} time steps as they come due (Ctrl-C to stop)")
        try:
            for ts in future:
                wait = (ts.to_pydatetime() - datetime.now(UTC)).total_seconds()
                if wait > 0:
                    time.sleep(wait)
                step = {node: f.loc[[ts]] for node, f in frames.items()}
                for message in to_messages(step, 1, sent_at=datetime.now(UTC)):
                    client.post("/ingest", message)
                log.info("posted %s", ts.isoformat())
        except KeyboardInterrupt:
            print("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())

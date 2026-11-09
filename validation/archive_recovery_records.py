"""Archive and release completed acceptance records through the public API."""

import argparse
import json
import re
import shutil
import time

from native_advanced import CLIENT, CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--keep-recent", type=int, default=12)
    arguments = parser.parse_args()
    if not 0 <= arguments.keep_recent <= 32:
        parser.error("--keep-recent must be between 0 and 32")
    if any(not re.fullmatch(r"[0-9a-f]{32}", item) for item in arguments.checkpoints):
        parser.error("Checkpoint IDs must come from checkpoint_create")
    root = CONFIG.workspace / ".cascadeur-mcp" / "recovery"
    destination = CONFIG.workspace / "validation" / f"recovery-archive-{time.time_ns()}"
    destination.mkdir()
    for identifier in arguments.checkpoints:
        source = root / "checkpoints" / (identifier + ".json")
        if source.is_file():
            shutil.copy2(source, destination / ("checkpoint-" + source.name))
            print(CLIENT.call("checkpoint_release", {"checkpoint_id": identifier})["result"], flush=True)
    records = [
        json.loads(path.read_text(encoding="utf-8")) for path in (root / "transactions").glob("*.json")
    ]
    records.sort(key=lambda item: item["created_at"])
    eligible = records[: max(0, len(records) - arguments.keep_recent)]
    for record in eligible:
        if record["status"] not in {"committed", "rolled_back", "recovered_to_checkpoint"}:
            continue
        identifier = record["request_id"]
        source = root / "transactions" / (identifier + ".json")
        shutil.copy2(source, destination / ("transaction-" + source.name))
        print(CLIENT.call("transaction_release", {"request_id": identifier})["result"], flush=True)
    print("ARTIFACT", destination, flush=True)


if __name__ == "__main__":
    main()

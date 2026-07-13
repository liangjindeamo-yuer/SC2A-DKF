from __future__ import annotations

import argparse
import os
import runpy
import sys
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    config = Path(args.config).resolve()
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    temp_dir = run_root / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    os.environ["WANDB_DIR"] = str(temp_dir)

    os.chdir(repo)
    sys.path.insert(0, str(repo))

    stdout_path = run_root / "train_stdout.log"
    stderr_path = run_root / "train_stderr.log"
    with stdout_path.open("a", encoding="utf-8", buffering=1) as stdout, stderr_path.open(
        "a", encoding="utf-8", buffering=1
    ) as stderr:
        sys.stdout = stdout
        sys.stderr = stderr
        print(f"[launcher] repo={repo}")
        print(f"[launcher] config={config}")
        print(f"[launcher] run_root={run_root}")
        print(f"[launcher] python={sys.executable}")
        sys.argv = ["main.py", "--config", str(config)]
        try:
            runpy.run_path(str(repo / "main.py"), run_name="__main__")
        except BaseException:
            traceback.print_exc(file=stderr)
            raise


if __name__ == "__main__":
    main()


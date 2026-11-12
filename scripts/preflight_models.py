"""
Pre-deploy gate: does every configured model exist, and does each platform key work? Free calls only (model lookups, nothing is generated).

  python scripts/preflight_models.py          # exit code 0 = all good, 1 = something would fail in production
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_preflight import run_preflight   # noqa: E402


def main() -> int:
    report = run_preflight(record=False)
    for provider, info in report["providers"].items():
        print(f"\n{provider}: key {info['key']}")
        for model_id, row in sorted(info["models"].items()):
            mark = {True: "ok     ", False: "MISSING", None: "?      "}[row["ok"]]
            print(f"  {mark} {model_id:34} {', '.join(row['roles'])}{'  <- ' + row['detail'] if row.get('detail') else ''}")
    print()
    if report["ok"]:
        print("PREFLIGHT OK")
        return 0
    print("PREFLIGHT FAILED:")
    for p in report["problems"]:
        print(f"  - {p['provider']} {p['model']}: {p['detail']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

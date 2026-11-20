"""
Pre-deploy gate, run on the target server (or its image) after installing dependencies:

  1. SDK contracts (offline): do the installed provider SDKs accept every argument the app sends? Catches a wrong SDK version, e.g.
     "Messages.stream() got an unexpected keyword argument 'output_config'" (2026-11-20). See src/sdk_contracts.py.
  2. Models and keys (free lookups, no generation): does every configured model exist, and does each platform key work?

  python scripts/preflight_models.py              # both; exit code 0 = all good, 1 = something would fail in production
  python scripts/preflight_models.py --sdk-only   # only the offline SDK check (CI, image build)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_preflight import run_preflight   # noqa: E402
from src.sdk_contracts import check_sdk_contracts   # noqa: E402


def sdk_check() -> bool:
    report = check_sdk_contracts()
    print("SDK versions: " + ", ".join(f"{k}={v}" for k, v in report["versions"].items()))
    if report["ok"]:
        print(f"SDK contracts OK ({report['checked']} request paths)")
        return True
    print("SDK CONTRACT MISMATCH (install requirements.lock):")
    for f in report["findings"]:
        print(f"  - {f['message']}")
    return False


def model_check() -> bool:
    report = run_preflight(record=False)
    for provider, info in report["providers"].items():
        print(f"\n{provider}: key {info['key']}")
        for model_id, row in sorted(info["models"].items()):
            mark = {True: "ok     ", False: "MISSING", None: "?      "}[row["ok"]]
            print(f"  {mark} {model_id:34} {', '.join(row['roles'])}{'  <- ' + row['detail'] if row.get('detail') else ''}")
    print()
    if report["ok"]:
        return True
    print("MODEL CHECK FAILED:")
    for p in report["problems"]:
        print(f"  - {p['provider']} {p['model']}: {p['detail']}")
    return False


def main(argv) -> int:
    ok = sdk_check()
    if "--sdk-only" not in argv:
        ok = model_check() and ok
    print("PREFLIGHT OK" if ok else "PREFLIGHT FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

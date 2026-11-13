"""A stand-in for worker.py that misbehaves on purpose, to prove sandbox.py contains it. Mode comes from the input file's contents."""
import json
import os
import sys
import time


def main(argv):
    mode = open(argv[2], "rb").read().decode()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import importlib.util
    spec = importlib.util.spec_from_file_location("real_worker", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src", "document_import", "worker.py"))
    real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real)
    if mode != "bigoutput_unlimited":
        real._limits()                              # the same limits the real worker applies before it parses anything
    out = argv[3]
    if mode == "env":
        with open(out, "w") as f:
            json.dump({"ok": True, "markdown": json.dumps({"env": dict(os.environ), "cwd": os.getcwd(), "argv0": sys.argv[0], "stdin": sys.stdin is None or sys.stdin.read(1) == ""}),
                       "title": "", "stats": {}, "warnings": []}, f)
    elif mode == "sleep":
        time.sleep(60)
    elif mode == "spin":
        while True:
            pass
    elif mode == "alloc":
        blocks = []
        while True:
            blocks.append(bytearray(256 * 1024 * 1024))
    elif mode == "crash":                           # dies without answering, but through an ordinary exit: a fatal signal (SIGABRT) makes macOS show a
        os._exit(70)                                # "Python quit unexpectedly" dialog for every run of the suite
    elif mode == "abort":                           # a real SIGABRT: Linux/CI only (see test_sandbox.py)
        os.abort()
    elif mode in ("bigoutput", "bigoutput_unlimited"):          # "bigoutput" runs under the output-size limit and is killed by SIGXFSZ: Linux/CI only
        with open(out, "w") as f:
            f.write("x" * (64 * 1024 * 1024))
    elif mode == "garbage":
        with open(out, "w") as f:
            f.write("not json {")
    elif mode == "spawn":                           # tries to leave a grandchild behind
        os.system("sleep 60 &")
        time.sleep(60)
    elif mode == "write_elsewhere":
        with open(out, "w") as f:
            json.dump({"ok": True, "markdown": str(os.path.exists(os.path.expanduser("~/.ssh"))), "title": "", "stats": {}, "warnings": []}, f)
    return 0


sys.exit(main(sys.argv))

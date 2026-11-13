"""
sandbox.py: a hostile or broken document costs one child process and nothing else.

The real worker is replaced by hostile_worker.py, which applies the real resource limits and then misbehaves in the way named by the input
(sleep, spin, allocate, crash, flood output, return garbage, leave a child behind). Each test checks the parent's answer, its timing, and that
no process is left running.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import src.document_import.sandbox as sandbox
from src.document_import.errors import ImportRefused
from tests.document_import.builders import column_page, make_docx, make_pdf, p

HOSTILE = str(Path(__file__).with_name("hostile_worker.py"))


@pytest.fixture
def hostile(monkeypatch):
    monkeypatch.setattr(sandbox, "_WORKER", HOSTILE)


def run(mode: str, **kw):
    return sandbox.run_converter("docx", mode.encode(), **kw)


def children_of_this_test() -> list:
    out = subprocess.run(["pgrep", "-f", "hostile_worker.py"], capture_output=True, text=True).stdout.split()
    return [pid for pid in out if pid]


class TestIsolation:
    def test_child_gets_an_empty_environment(self, hostile, monkeypatch):
        monkeypatch.setenv("DEFAULT_OPENAI_API_KEY", "sk-must-not-leak")
        monkeypatch.setenv("REDIS_URL", "redis://must-not-leak")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-leak")
        seen = json.loads(run("env").markdown)
        blob = json.dumps(seen["env"])
        assert "must-not-leak" not in blob and "sk-" not in blob
        names = {n for n in seen["env"] if not n.startswith("__CF")}              # macOS adds __CF_USER_TEXT_ENCODING to every process
        assert names <= {"LANG", "LC_CTYPE", "DOC_IMPORT_MEMORY_BYTES", "DOC_IMPORT_CPU_SECONDS", "DOC_IMPORT_OUTPUT_BYTES", "DOC_IMPORT_MAX_CHARS",
                                    "DOC_IMPORT_MAX_PAGES", "DOC_IMPORT_TIME_BUDGET"}

    def test_child_works_in_a_private_directory_and_has_no_stdin(self, hostile):
        seen = json.loads(run("env").markdown)
        assert "docimport-" in seen["cwd"]
        assert seen["stdin"] is True
        assert not os.path.exists(seen["cwd"])                      # removed afterwards

    def test_extra_env_is_passed_but_only_what_the_caller_names(self, hostile):
        seen = json.loads(sandbox.run_converter("html", b"env", extra_env={"DOC_IMPORT_BASE_URL": "https://x.test"}).markdown)
        assert seen["env"]["DOC_IMPORT_BASE_URL"] == "https://x.test"


class TestContainment:
    def test_a_hang_is_killed_at_the_timeout(self, hostile):
        t = time.perf_counter()
        with pytest.raises(ImportRefused) as e:
            run("sleep", timeout=1.5)
        assert e.value.code == "import_timeout" and e.value.status == 422
        assert time.perf_counter() - t < 6
        time.sleep(0.3)
        assert not children_of_this_test()

    def test_a_cpu_spin_is_stopped(self, hostile):
        t = time.perf_counter()
        with pytest.raises(ImportRefused) as e:
            run("spin", timeout=1.0)
        assert e.value.code in ("import_timeout", "too_complex")
        assert time.perf_counter() - t < 8
        time.sleep(0.3)
        assert not children_of_this_test()

    def test_a_child_that_exits_without_answering_is_a_clean_refusal(self, hostile):
        with pytest.raises(ImportRefused) as e:
            run("crash")
        assert e.value.status in (413, 422)
        assert "Traceback" not in e.value.message and "abort" not in e.value.message.lower()

    @pytest.mark.skipif(sys.platform == "darwin", reason="a fatal signal makes macOS show a 'Python quit unexpectedly' dialog on every run")
    def test_a_child_killed_by_a_signal_is_a_clean_refusal(self, hostile):
        with pytest.raises(ImportRefused) as e:
            run("abort")
        assert e.value.status in (413, 422) and "Traceback" not in e.value.message

    def test_an_oversized_answer_is_refused_by_the_parent(self, hostile):
        with pytest.raises(ImportRefused) as e:
            run("bigoutput_unlimited")                                  # no child-side limit: the parent's own size check must catch it
        assert e.value.status in (413, 422)

    @pytest.mark.skipif(sys.platform == "darwin", reason="SIGXFSZ from the file-size limit would be reported as a crash on macOS")
    def test_a_flood_of_output_is_cut_off_by_the_child_file_size_limit(self, hostile):
        with pytest.raises(ImportRefused) as e:
            run("bigoutput")
        assert e.value.status in (413, 422)

    def test_garbage_output_is_a_refusal_not_an_exception(self, hostile):
        with pytest.raises(ImportRefused) as e:
            run("garbage")
        assert e.value.code == "unreadable"

    def test_a_grandchild_does_not_survive_the_kill(self, hostile):
        with pytest.raises(ImportRefused):
            run("spawn", timeout=1.5)
        time.sleep(0.5)
        left = subprocess.run(["pgrep", "-f", "sleep 60"], capture_output=True, text=True).stdout.split()
        for pid in left:                                              # clean up if it did, then fail
            os.kill(int(pid), 9)
        assert not left

    @pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is not enforced on macOS")
    def test_memory_limit_stops_an_allocation_bomb(self, hostile):
        with pytest.raises(ImportRefused) as e:
            run("alloc", memory_bytes=512 * 1024 * 1024, timeout=20)
        assert e.value.status in (413, 422)


class TestRealWorker:
    def test_docx_and_pdf_round_trip_through_the_child(self):
        d = sandbox.run_converter("docx", make_docx([p("Title", "Title"), p("Hello")]))
        assert d.markdown.startswith("# Title") and "Hello" in d.markdown
        f = sandbox.run_converter("pdf", make_pdf([column_page([("B", "Heading", 20), ("R", "Body text here. " * 15, 11)])]))
        assert f.markdown.startswith("# Heading") and f.stats["pages"] == 1

    def test_a_refusal_keeps_its_code_and_status(self):
        with pytest.raises(ImportRefused) as e:
            sandbox.run_converter("pdf", b"%PDF-1.4 junk")
        assert (e.value.status, e.value.code) == (422, "pdf_unreadable")

    def test_html_kind(self):
        r = sandbox.run_converter("html", b"<h1>Hi</h1><p>there</p>", extra_env={"DOC_IMPORT_BASE_URL": "https://x.test/"})
        assert r.markdown.strip().endswith("there") and r.markdown.startswith("# Hi")

    def test_the_real_worker_applies_its_resource_limits_before_it_parses(self, tmp_path, monkeypatch):
        """In-process, so nothing is killed by a signal: main() must call _limits() before it opens the input."""
        import json
        import src.document_import.worker as worker
        order = []
        real_open = open

        def spy_open(path, *a, **kw):
            if str(path).endswith("input.docx"):
                order.append("read input")
            return real_open(path, *a, **kw)
        monkeypatch.setattr(worker, "_limits", lambda: order.append("limits"))
        monkeypatch.setattr("builtins.open", spy_open)
        source, target = tmp_path / "input.docx", tmp_path / "result.json"
        source.write_bytes(make_docx([p("Hello")]))
        assert worker.main(["worker.py", "docx", str(source), str(target)]) == 0
        monkeypatch.undo()
        assert order[:2] == ["limits", "read input"]
        assert json.loads(target.read_text())["ok"] is True

    def test_the_limits_come_from_the_environment_the_parent_sets(self):
        """_limits() reads the parent's variables and applies them (checked in a throw-away process that only reads them back, so nothing is killed)."""
        import subprocess
        code = ("import importlib.util, resource, sys\n"
                f"spec = importlib.util.spec_from_file_location('w', {sandbox._WORKER!r}); w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)\n"
                "w._limits()\n"
                "print(resource.getrlimit(resource.RLIMIT_CORE)[0], resource.getrlimit(resource.RLIMIT_NOFILE)[0], resource.getrlimit(resource.RLIMIT_CPU)[0], "
                "resource.getrlimit(resource.RLIMIT_FSIZE)[0])")
        out = subprocess.run([sys.executable, "-I", "-c", code], env={"DOC_IMPORT_CPU_SECONDS": "7", "DOC_IMPORT_OUTPUT_BYTES": "123456"},
                             capture_output=True, text=True, timeout=30).stdout.split()
        assert out == ["0", "64", "7", "123456"]

    def test_unknown_kind_is_refused(self):
        with pytest.raises(ImportRefused):
            sandbox.run_converter("exe", b"x")

    def test_the_parent_process_environment_is_untouched(self, monkeypatch):
        monkeypatch.setenv("SOME_SECRET", "still-here")
        sandbox.run_converter("docx", make_docx([p("Hello")]))
        assert os.environ["SOME_SECRET"] == "still-here"

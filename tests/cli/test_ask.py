"""Tests for Phase 10: agy-ask CLI Worker (Section 16)."""
import json
import pytest
from pathlib import Path

from agy_bridge.cli.ask import run_ask, CLIResult


class FakeSubprocessRunner:
    def __init__(self, stdout: str = "Fake CLI response", stderr: str = "", exit_code: int = 0, times_out: bool = False):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.times_out = times_out
        self.invocations = []

    def run(self, cmd, input_bytes=None, timeout=None):
        self.invocations.append({"cmd": cmd, "input": input_bytes, "timeout": timeout})
        if self.times_out:
            raise TimeoutError("Execution timed out")
        return self.exit_code, self.stdout, self.stderr


def test_cli_requires_exactly_one_prompt_source():
    runner = FakeSubprocessRunner()

    # Neither provided -> exit code 2
    res1 = run_ask(prompt_file=None, use_stdin=False, runner=runner)
    assert res1.exit_code == 2
    assert "exactly one" in (res1.error or "").lower()

    # Both provided -> exit code 2
    res2 = run_ask(prompt_file="some_file.md", use_stdin=True, runner=runner)
    assert res2.exit_code == 2
    assert "exactly one" in (res2.error or "").lower()


def test_cli_success_with_stdin_prompt(tmp_path: Path):
    runner = FakeSubprocessRunner(stdout="Calculated answer is 42", exit_code=0)
    out_file = tmp_path / "result.json"

    res = run_ask(
        prompt_text="What is 6 * 7?",
        use_stdin=True,
        output_path=str(out_file),
        model="gemini-3.8-flash",
        runner=runner,
    )

    assert res.exit_code == 0
    assert res.status == "success"
    assert res.text == "Calculated answer is 42"
    assert res.requested_model == "gemini-3.8-flash"
    assert out_file.exists()

    payload = json.loads(out_file.read_text())
    assert payload["schema_version"] == 1
    assert payload["status"] == "success"
    assert payload["text"] == "Calculated answer is 42"


def test_cli_handles_timeout():
    runner = FakeSubprocessRunner(times_out=True)
    res = run_ask(
        prompt_text="Take forever",
        use_stdin=True,
        timeout_seconds=5,
        runner=runner,
    )

    assert res.exit_code == 124
    assert res.status == "timeout"
    assert "timed out" in (res.error or "").lower()


def test_cli_classifies_quota_error():
    runner = FakeSubprocessRunner(
        stdout="",
        stderr="Error 429: Resource has been exhausted (e.g. check quota)",
        exit_code=1,
    )
    res = run_ask(
        prompt_text="Run job",
        use_stdin=True,
        runner=runner,
    )

    assert res.exit_code == 21
    assert res.status == "quota_exhausted"


def test_cli_classifies_auth_error():
    runner = FakeSubprocessRunner(
        stdout="",
        stderr="Error: User is not logged in. Please run gemini auth.",
        exit_code=1,
    )
    res = run_ask(
        prompt_text="Run job",
        use_stdin=True,
        runner=runner,
    )

    assert res.exit_code == 20
    assert res.status == "auth_required"

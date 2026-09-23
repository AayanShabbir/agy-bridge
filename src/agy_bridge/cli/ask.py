"""agy-ask CLI Worker (Section 16).

Wraps /opt/homebrew/bin/gemini as a separate, bounded execution path.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Tuple


@dataclass
class CLIResult:
    schema_version: int = 1
    status: str = "success"
    backend: str = "gemini-cli"
    cli_version: str = "observed-version"
    requested_model: str = "gemini-3.8-flash"
    served_model: Optional[str] = None
    text: Optional[str] = None
    duration_ms: int = 0
    error: Optional[str] = None
    exit_code: int = 0


class RealSubprocessRunner:
    def run(
        self,
        cmd: list[str],
        input_bytes: Optional[bytes] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[int, str, str]:
        proc = subprocess.run(
            cmd,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return (
            proc.returncode,
            proc.stdout.decode("utf-8", errors="replace"),
            proc.stderr.decode("utf-8", errors="replace"),
        )


def run_ask(
    prompt_file: Optional[str] = None,
    use_stdin: bool = False,
    prompt_text: Optional[str] = None,
    output_path: Optional[str] = None,
    model: str = "gemini-3.8-flash",
    timeout_seconds: Optional[int] = 60,
    executable: str = "/opt/homebrew/bin/gemini",
    runner: Optional[Any] = None,
) -> CLIResult:
    """Execute gemini CLI with bounded execution and normalized result schema."""
    start_time = time.monotonic()
    runner = runner or RealSubprocessRunner()

    # Validation: Exactly one prompt source
    sources = 0
    if prompt_file:
        sources += 1
    if use_stdin:
        sources += 1

    if sources != 1:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        return CLIResult(
            status="invocation_error",
            requested_model=model,
            duration_ms=duration_ms,
            error="Exactly one prompt source must be specified (--prompt-file or --stdin)",
            exit_code=2,
        )

    # Prepare prompt bytes
    input_bytes = None
    if use_stdin:
        input_bytes = (prompt_text or "").encode("utf-8")
    elif prompt_file:
        try:
            with open(prompt_file, "rb") as f:
                input_bytes = f.read()
        except OSError as exc:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return CLIResult(
                status="invocation_error",
                requested_model=model,
                duration_ms=duration_ms,
                error=f"Failed to read prompt file {prompt_file!r}: {exc}",
                exit_code=2,
            )

    cmd = [executable, "--model", model]

    try:
        rc, stdout, stderr = runner.run(cmd, input_bytes=input_bytes, timeout=timeout_seconds)
    except TimeoutError:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        res = CLIResult(
            status="timeout",
            requested_model=model,
            duration_ms=duration_ms,
            error=f"Execution timed out after {timeout_seconds}s",
            exit_code=124,
        )
        if output_path:
            _write_output(output_path, res)
        return res
    except Exception as exc:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        res = CLIResult(
            status="invocation_error",
            requested_model=model,
            duration_ms=duration_ms,
            error=f"Subprocess execution failed: {exc}",
            exit_code=2,
        )
        if output_path:
            _write_output(output_path, res)
        return res

    duration_ms = int((time.monotonic() - start_time) * 1000)

    # Classify stderr / exit code
    combined = (stdout + "\n" + stderr).lower()
    if rc == 0:
        res = CLIResult(
            status="success",
            requested_model=model,
            text=stdout,
            duration_ms=duration_ms,
            exit_code=0,
        )
    elif "quota" in combined or "429" in combined or "exhausted" in combined:
        res = CLIResult(
            status="quota_exhausted",
            requested_model=model,
            duration_ms=duration_ms,
            error=stderr.strip() or stdout.strip(),
            exit_code=21,
        )
    elif "not logged in" in combined or "auth" in combined or "login" in combined:
        res = CLIResult(
            status="auth_required",
            requested_model=model,
            duration_ms=duration_ms,
            error=stderr.strip() or stdout.strip(),
            exit_code=20,
        )
    elif rc == 130:
        res = CLIResult(
            status="interrupted",
            requested_model=model,
            duration_ms=duration_ms,
            error="Interrupted",
            exit_code=130,
        )
    else:
        res = CLIResult(
            status="upstream_failure",
            requested_model=model,
            duration_ms=duration_ms,
            error=stderr.strip() or stdout.strip(),
            exit_code=22,
        )

    if output_path:
        _write_output(output_path, res)

    return res


def _write_output(output_path: str, result: CLIResult) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp_path = p.with_suffix(".tmp")
    data = asdict(result)
    temp_path.write_text(json.dumps(data, indent=2))
    temp_path.replace(p)

import subprocess
import sys
from unittest.mock import patch

def test_cli_runner_local_mode(tmp_path):
    """Test that CLI runner works in local mode with mocked output."""
    # Path to cli_runner.py
    cli_script = "src/cli_runner.py"

    test_input = "This is a test clinical note."
    expected_summary = "Mocked local summary."

    # Patch Summarizer.summarize to avoid calling the real model
    with patch("src.cli_runner.Summarizer.summarize", return_value=expected_summary):
        result = subprocess.run(
            [sys.executable, cli_script, "--mode", "local", "--text", test_input],
            capture_output=True,
            text=True
        )

    assert result.returncode == 0
    assert expected_summary in result.stdout


def test_cli_runner_gemini_mode(tmp_path):
    """Test that CLI runner works in Gemini mode with mocked output."""
    cli_script = "src/cli_runner.py"

    test_input = "This is another test clinical note."
    expected_summary = "Mocked gemini summary."

    with patch("src.cli_runner.Summarizer.summarize", return_value=expected_summary):
        result = subprocess.run(
            [sys.executable, cli_script, "--mode", "gemini", "--text", test_input],
            capture_output=True,
            text=True
        )

    assert result.returncode == 0
    assert expected_summary in result.stdout

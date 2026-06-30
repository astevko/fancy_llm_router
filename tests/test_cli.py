"""Tests for CLI commands."""

import json
import logging
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from fancy_llm_router.cli.main import cli
from tests.conftest import MOCK_CONFIG


def _extract_json(output: str) -> dict:
    """Parse JSON from CLI output that may include log lines."""
    start = output.find("{")
    assert start != -1, output
    return json.loads(output[start:])


@pytest.fixture(autouse=True)
def quiet_router_logs():
    logging.getLogger("fancy_llm_router").setLevel(logging.CRITICAL)


@pytest.fixture
def mock_config_file(tmp_path: Path) -> Path:
    path = tmp_path / "mock.yaml"
    path.write_text(yaml.safe_dump(MOCK_CONFIG, sort_keys=False))
    return path


@pytest.fixture
def runner():
    return CliRunner()


class TestCLI:
    def test_route_json_output(self, runner, mock_config_file):
        result = runner.invoke(
            cli,
            ["-c", str(mock_config_file), "route", "-p", "hello", "--json"],
        )
        assert result.exit_code == 0, result.output
        data = _extract_json(result.output)
        assert "selected_deployment" in data
        assert "candidates" in data

    def test_route_table_output(self, runner, mock_config_file):
        result = runner.invoke(
            cli,
            ["-c", str(mock_config_file), "route", "-p", "hello"],
        )
        assert result.exit_code == 0, result.output
        assert "Routing Decision" in result.output
        assert "Selected Deployment" in result.output

    def test_complete_json_with_mock_provider(self, runner, mock_config_file):
        result = runner.invoke(
            cli,
            [
                "-c",
                str(mock_config_file),
                "complete",
                "--prompt",
                "what is the capital of france?",
                "--strategy",
                "cost_optimized",
                "--max-tokens",
                "32",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        data = _extract_json(result.output)
        assert "response" in data
        assert data["response"]["usage"]["total_tokens"] > 0
        assert "routing_decision" in data

    def test_complete_accepts_positional_prompt(self, runner, mock_config_file):
        result = runner.invoke(
            cli,
            ["-c", str(mock_config_file), "complete", "hello", "--json"],
        )
        assert result.exit_code == 0, result.output
        assert _extract_json(result.output)["response"]["choices"]

    def test_list_models(self, runner, mock_config_file):
        result = runner.invoke(cli, ["-c", str(mock_config_file), "list-models"])
        assert result.exit_code == 0, result.output
        assert "Available Deployments" in result.output
        assert "qwen@nebius" in result.output or "qwen@o" in result.output
        assert "mock-fa" in result.output

"""Tests for YAML config loading and deployment registration."""

from fancy_llm_router.core.config_loader import build_router_from_config
from tests.conftest import MOCK_CONFIG


class TestConfigLoader:
    def test_builds_deployment_keys(self, mock_router):
        ids = mock_router.list_models()
        assert "mock-fast@mock" in ids
        assert "qwen@nebius" in ids
        assert "qwen@ollama" in ids

    def test_logical_model_and_source(self, mock_router):
        nebius = mock_router.get_model_info("qwen@nebius")
        ollama = mock_router.get_model_info("qwen@ollama")

        assert nebius.logical_model == "Qwen/Qwen3-32B"
        assert ollama.logical_model == "Qwen/Qwen3-32B"
        assert nebius.source == "nebius"
        assert ollama.source == "ollama"
        assert nebius.model_id == "Qwen/Qwen3-32B"
        assert ollama.model_id == "qwen3:32b"

    def test_skips_disabled_deployments(self):
        cfg = {
            "models": {
                "off@mock": {
                    "model": "off",
                    "provider": "mock",
                    "model_id": "off",
                    "enabled": False,
                },
                "on@mock": {
                    "model": "on",
                    "provider": "mock",
                    "model_id": "on",
                    "enabled": True,
                },
            }
        }
        router = build_router_from_config(cfg)
        assert router.list_models() == ["on@mock"]

    def test_env_var_expansion(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TEST_LLM_KEY", "secret-value")
        path = tmp_path / "cfg.yaml"
        path.write_text(
            "models:\n"
            "  x@mock:\n"
            "    model: x\n"
            "    provider: mock\n"
            "    model_id: x\n"
            "    api_key: ${TEST_LLM_KEY}\n"
            "    enabled: true\n"
        )
        from fancy_llm_router.core.config_loader import create_router

        router = create_router(path)
        provider = router._providers["x@mock"]
        assert provider.api_key == "secret-value"

    def test_loads_api_key_from_dotenv_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TEST_DOTENV_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("TEST_DOTENV_KEY=from-dotenv-file\n")
        path = tmp_path / "cfg.yaml"
        path.write_text(
            "models:\n"
            "  x@mock:\n"
            "    model: x\n"
            "    provider: mock\n"
            "    model_id: x\n"
            "    api_key: ${TEST_DOTENV_KEY}\n"
            "    enabled: true\n"
        )
        from fancy_llm_router.core.config_loader import create_router

        router = create_router(path)
        assert router._providers["x@mock"].api_key == "from-dotenv-file"

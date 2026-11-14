from pathlib import Path

ROOT = Path(__file__).parents[1]
SERVICE = (ROOT / "modal/serve.py").read_text(encoding="utf-8")


def test_service_pins_the_accepted_release() -> None:
    assert "finance_agent.release" in SERVICE
    assert "ADAPTER_REVISION" in SERVICE
    assert "modal.fastapi_endpoint" in SERVICE
    assert "def health" in SERVICE
    assert "def version" in SERVICE


def test_service_does_not_embed_credentials() -> None:
    assert "HF_PRIMARY_TOKEN" not in SERVICE
    assert "HF_SECONDARY_TOKEN" not in SERVICE
    assert "hf-read-token" in SERVICE

"""Verify local Anthropic and Hugging Face credentials without exposing secrets.

This script performs read-only checks. It does not start Jobs, create repositories,
download model weights, or make a paid text-generation request.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv(override=False)


MODEL_ID = os.getenv("HF_MODEL_ID", "CohereLabs/tiny-aya-global")


def check_huggingface(label: str, token_name: str, expected_username: str) -> bool:
    token = os.getenv(token_name)
    if not token:
        print(f"[FAIL] {label}: {token_name} is not set")
        return False

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        account = api.whoami()
        username = account.get("name") or account.get("user", {}).get("name")
        if expected_username and (
            not username or username.casefold() != expected_username.casefold()
        ):
            print(
                f"[FAIL] {label}: authenticated as {username!r}, "
                f"expected {expected_username!r}"
            )
            return False

        api.model_info(MODEL_ID, token=token)
        print(f"[PASS] {label}: authenticated as {username}; model access confirmed")
        return True
    except Exception as error:  # Keep the diagnostic readable without leaking tokens.
        print(f"[FAIL] {label}: {type(error).__name__}: {error}")
        return False


def check_anthropic() -> bool:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[FAIL] Anthropic: ANTHROPIC_API_KEY is not set")
        return False

    try:
        from anthropic import Anthropic

        client = Anthropic()
        models = client.models.list(limit=1)
        print(f"[PASS] Anthropic: API key accepted; {len(models.data)} model result returned")
        return True
    except Exception as error:  # Do not print credentials or request headers.
        print(f"[FAIL] Anthropic: {type(error).__name__}: {error}")
        return False


def main() -> int:
    checks = [
        check_huggingface(
            "HF primary",
            "HF_PRIMARY_TOKEN",
            os.getenv("HF_PRIMARY_USERNAME", "saipavankumar22"),
        ),
        check_huggingface(
            "HF secondary",
            "HF_SECONDARY_TOKEN",
            os.getenv("HF_SECONDARY_USERNAME", "avengeryt22"),
        ),
        check_anthropic(),
    ]
    print(f"\nModel target: {MODEL_ID}")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Create the initial Hugging Face repositories for the finance agent."""

import os
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(override=False)


def required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"{name} is not set in the environment or .env")
    return value


primary_token = required("HF_PRIMARY_TOKEN")
secondary_token = required("HF_SECONDARY_TOKEN")
primary_user = required("HF_PRIMARY_USERNAME")
secondary_user = required("HF_SECONDARY_USERNAME")

primary_api = HfApi(token=primary_token)
secondary_api = HfApi(token=secondary_token)

repositories = [
    (
        primary_api,
        f"{primary_user}/finance-agent",
        "model",
        False,
        {},
    ),
    (
        secondary_api,
        f"{secondary_user}/finance-agent-staging",
        "model",
        True,
        {},
    ),
    (
        primary_api,
        f"{primary_user}/finance-agent-demo",
        "space",
        False,
        {"space_sdk": "static"},
    ),
]

failures = 0

for api, repo_id, repo_type, private, extra in repositories:
    try:
        repo = api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
            **extra,
        )
        print(f"[PASS] {repo_type}: {repo}")
    except Exception as error:
        failures += 1
        print(f"[FAIL] {repo_type} {repo_id}: {type(error).__name__}: {error}")

print("\nRepository setup complete.")
if failures:
    sys.exit(1)

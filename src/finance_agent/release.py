"""Release metadata shared by local tooling and deployment code."""

from __future__ import annotations

MODEL_ID = "CohereLabs/tiny-aya-global"
RELEASE_NAME = "ic7"
RELEASE_REVISION = "5ae2ee8"
ADAPTER_STAGING = f"avengeryt22/finance-agent-staging@{RELEASE_REVISION}"
ADAPTER_PRIMARY = "SaiPavankumar22/finance-agent@0366a47"
ADAPTER_REPO = "avengeryt22/finance-agent-staging"
ADAPTER_REVISION = RELEASE_REVISION

PUBLIC_QUERY_ENDPOINT = (
    "https://devisettisaipavankumar--finance-agent-ws5-demo-financeag-dc3c3e."
    "us-east.modal.run"
)
PUBLIC_VERSION_ENDPOINT = (
    "https://devisettisaipavankumar--finance-agent-ws5-demo-financeag-66c484."
    "us-east.modal.run"
)

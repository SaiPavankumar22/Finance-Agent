# Deployment

The public service is defined in [`../modal/serve.py`](../modal/serve.py).
That path is retained as the compatibility entry point used by the existing
Modal deployment:

```powershell
modal deploy modal/serve.py
```

The service reads its model and adapter identity from
`finance_agent.release`. The accepted release is pinned to:

- model: `CohereLabs/tiny-aya-global`
- staging adapter: `avengeryt22/finance-agent-staging@5ae2ee8`
- primary adapter: `SaiPavankumar22/finance-agent@0366a47`

The adapter is loaded from the Hub with the `hf-read-token` Modal secret.
Secrets never belong in source files, the Space, or the browser.


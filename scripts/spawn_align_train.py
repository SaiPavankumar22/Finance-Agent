"""Schedule an AlignTune training run on the deployed app and exit.

    modal run scripts/spawn_align_train.py --output aligntune-r32-e4-v3

The call belongs to the deployed `finance-agent-aligntune` app, so closing this
shell - or losing the connection - cannot cancel a two-hour run. Progress is in
`modal app logs finance-agent-aligntune`.
"""

from __future__ import annotations

import modal

app = modal.App("align-train-client")
TRAIN_APP = "finance-agent-aligntune"
TRAIN_FUNCTION = "run"


@app.local_entrypoint()
def main(output: str = "aligntune-r32-e4-v3", full: bool = True, steps: int = 20) -> None:
    function = modal.Function.from_name(TRAIN_APP, TRAIN_FUNCTION)
    call = function.spawn(max_steps=-1 if full else steps, output=output)
    print(f"spawned {call.object_id} -> /out/{output}")

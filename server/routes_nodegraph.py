"""Node-graph playground routes -- dev/testing tab, isolated from the
production config/training path (see docs/node_architecture_refactor_plan.md).

This is a development scaffold, not a production feature yet: it only reads
and displays introspected class metadata. It doesn't build, run, or persist
any graph, and doesn't touch config.py, config_model.py, or the training
launch path in any way. Safe to iterate on without any risk to existing
training runs.
"""

from fastapi import APIRouter, HTTPException

from .nodegraph_introspect import introspect_optimizers, node_info_to_dict

router = APIRouter(prefix="/nodegraph")


@router.get("/optimizers")
async def list_optimizer_nodes():
    """First proof-of-concept endpoint: introspected optimizers.py classes,
    rendered as node info. See nodegraph_introspect.py for why this
    introspects the real classes rather than a hand-maintained list.
    """
    try:
        infos = introspect_optimizers()
    except Exception as e:
        # Most likely cause: torch not importable in this environment.
        # Surface it plainly rather than a bare 500 -- this endpoint is a
        # dev tool, the person looking at it wants to know why, not just
        # that something failed.
        raise HTTPException(
            status_code=500,
            detail=f"Could not introspect optimizers.py ({type(e).__name__}: {e}). "
                   f"This usually means torch isn't importable in the current "
                   f"environment -- the training subprocess environment is "
                   f"expected to have it; the server process may not.",
        )
    return {"nodes": [node_info_to_dict(i) for i in infos]}

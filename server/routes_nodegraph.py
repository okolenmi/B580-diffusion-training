"""Node-graph playground routes -- dev/testing tab, isolated from the
production config/training path (see docs/node_architecture_refactor_plan.md).

This is a development scaffold, not a production feature yet: it only reads
and displays introspected class metadata. It doesn't build, run, or persist
any graph, and doesn't touch config.py, config_model.py, or the training
launch path in any way. Safe to iterate on without any risk to existing
training runs.
"""

from fastapi import APIRouter, HTTPException

from .nodegraph_introspect import introspect_optimizer_nodes, node_info_to_dict

router = APIRouter(prefix="/nodegraph")


@router.get("/optimizers")
async def list_optimizer_nodes():
    """Reads declared contracts directly off nodes/optimizer/'s real Node
    classes -- see nodegraph_introspect.py's introspect_optimizer_nodes()
    and docs/nodes_package_design.md. This replaced an earlier version that
    guessed ports from core.optimizers.py's constructor signatures; that
    approach is still available (introspect_legacy_class(), same module)
    for any future domain not yet migrated into nodes/.
    """
    try:
        infos = introspect_optimizer_nodes()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not introspect nodes/optimizer/ ({type(e).__name__}: {e}).",
        )
    return {"nodes": [node_info_to_dict(i) for i in infos]}

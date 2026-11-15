import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    reference: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    model_id: Optional[str] = None
    wait_for_result: Optional[bool] = True


class QueryResponse(BaseModel):
    task_id: Optional[str] = None
    model_id: str
    response: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def get_orchestrator(request: Request):
    """The orchestrator that run_api's lifespan put on the app state."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return orchestrator


@router.post("/inference", response_model=QueryResponse)
async def inference_endpoint(
    request: QueryRequest, orchestrator=Depends(get_orchestrator)
):
    """Process inference request"""
    try:
        result = await orchestrator.process_query(
            query_text=request.query,
            reference=request.reference,
            metadata=request.metadata,
            model_id=request.model_id,
            wait_for_result=request.wait_for_result,
        )
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

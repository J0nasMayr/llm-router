import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")
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


def get_orchestrator():
    from run_api import orchestrator

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

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

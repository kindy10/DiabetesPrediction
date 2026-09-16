from fastapi import APIRouter


router = APIRouter(
    tags=["System"]
)


@router.get(
    "/",
    summary="API information"
)
def root():
    return {
        "message": "Diabetes Prediction API",
        "status": "running"
    }


@router.get(
    "/health",
    summary="Check API health"
)
def health():
    return {
        "status": "healthy"
    }
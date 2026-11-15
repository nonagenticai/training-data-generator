"""API routers for the training data generator service."""

from routers.batches import router as batches_router
from routers.triggers import router as triggers_router

__all__ = [
    "batches_router",
    "triggers_router",
]

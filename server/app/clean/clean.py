from pydantic import BaseModel
from typing import Dict, Optional


class CleanConnection(BaseModel):
    entities: Dict[str, int]
    groups: Dict[str, str]
    counts: Dict[str, int]
    metrics: Dict[str, float]
    stats: Dict[str, Optional[float]]

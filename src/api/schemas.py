from pydantic import BaseModel
from typing import Dict, Optional, List, Union

class PredictionRequest(BaseModel):
    columns: Dict[str, List[Optional[Union[float,str]]]]

class PredictionResponse(BaseModel):
    predictions: List[float]



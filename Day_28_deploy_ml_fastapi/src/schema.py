from pydantic import BaseModel

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float

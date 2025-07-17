from pydantic import BaseModel
from typing import Optional

class HyroxModelRequest(BaseModel):
    # optionals for identification
    race_id: Optional[str] = None
    name: Optional[str] = None
    nationality: Optional[str] = None
    # required context information
    gender: str
    age: int
    # required performance data
    run_1: float
    run_2: float
    run_3: float
    run_4: float
    run_5: float
    run_6: float
    run_7: float
    run_8: float
    work_1: float
    work_2: float
    work_3: float
    work_4: float
    work_5: float
    work_6: float
    work_7: float
    work_8: float
    roxzone_1: float
    roxzone_2: float
    roxzone_3: float
    roxzone_4: float
    roxzone_5: float
    roxzone_6: float
    roxzone_7: float
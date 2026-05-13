from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "relay"
PILOT_PATH = Path(__file__).resolve().parent / "data" / "pilot.json"
COMPANIES = {
    "AMD": (2488, "Advanced Micro Devices"),
    "NVDA": (1045810, "NVIDIA"),
    "INTC": (50863, "Intel"),
    "QCOM": (804328, "Qualcomm"),
    "TXN": (97476, "Texas Instruments"),
    "MU": (723125, "Micron Technology"),
}


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: date = date(2017, 1, 1)
    end: date = Field(default_factory=lambda: datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1))
    horizon_days: int = Field(default=5, ge=1, le=20)
    cost_bps: float = Field(default=20, ge=0, le=200)
    min_train_rows: int = Field(default=100, ge=50, le=10000)

    @model_validator(mode="after")
    def dates_are_ordered(self):
        if self.start >= self.end:
            raise ValueError("start must precede end")
        if self.end >= datetime.now(ZoneInfo("America/New_York")).date():
            raise ValueError("end must precede today; current-session daily bars may be incomplete")
        return self

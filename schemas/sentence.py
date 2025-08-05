from pydantic import BaseModel
from typing import Optional


class SentenceBase(BaseModel):
    ID_sen: str
    Left: Optional[str] = ""
    Center: Optional[str] = ""
    Right: Optional[str] = ""


class SentenceCreate(SentenceBase):
    pass


class SentenceRead(SentenceBase):
    class Config:
        orm_mode = True

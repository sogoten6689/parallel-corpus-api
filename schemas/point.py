from pydantic import BaseModel


class PointBase(BaseModel):
    StartPos: int
    EndPos: int
    SentenceID: str


class PointCreate(PointBase):
    pass


class PointRead(PointBase):
    id: int

    class Config:
        orm_mode = True

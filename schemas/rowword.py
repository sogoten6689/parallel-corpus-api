from pydantic import BaseModel
from typing import Optional

class RowWordBase(BaseModel):
    ID: str
    ID_sen: str
    Word: str
    Lemma: str
    Links: str
    # Morph: str or null
    Morph: Optional[str]
    POS: str
    Phrase: str
    Grm: str
    NER: str
    Semantic: str
    Lang_code: str


class RowWordCreate(RowWordBase):
    pass

class RowWordRead(RowWordBase):
    class Config:
        orm_mode = True

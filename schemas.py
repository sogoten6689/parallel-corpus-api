from pydantic import BaseModel

class RowWordBase(BaseModel):
    ID: str
    ID_sen: str
    Word: str
    Lemma: str
    Links: str
    Morph: str
    POS: str
    Phrase: str
    Grm: str
    NER: str
    Semantic: str

class RowWordCreate(RowWordBase):
    pass

class RowWordRead(RowWordBase):
    class Config:
        orm_mode = True

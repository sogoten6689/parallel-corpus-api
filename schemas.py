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

class RowWordUpdate(BaseModel):
    ID: str
    ID_sen: str
    Word: str | None = None
    Lemma: str | None = None
    Links: str | None = None
    Morph: str | None = None
    POS: str | None = None
    Phrase: str | None = None
    Grm: str | None = None
    NER: str | None = None
    Semantic: str | None = None

class CorpusBase(BaseModel):
    sentence_id: str
    language: str
    text: str
    translation: str | None = None
    meta_data: str | None = None

class CorpusCreate(CorpusBase):
    pass

class CorpusRead(CorpusBase):
    class Config:
        from_attributes = True

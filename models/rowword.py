from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String
from .base import Base

class RowWord(Base):
    __tablename__ = "row_words"

    ID = Column(String, primary_key=True, index=True)
    ID_sen = Column(String, index=True)
    Word = Column(String)
    Lemma = Column(String)
    Links = Column(String)
    Morph = Column(String)
    POS = Column(String)
    Phrase = Column(String)
    Grm = Column(String)
    NER = Column(String)
    Semantic = Column(String)
    Lang_code = Column(String)

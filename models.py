from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Text, Integer

Base = declarative_base()

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

class Corpus(Base):
    __tablename__ = "corpus"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    sentence_id = Column(String, index=True)
    language = Column(String)
    text = Column(Text)
    translation = Column(Text, nullable=True)
    meta_data = Column(Text, nullable=True)

from sqlalchemy import Column, Integer, String
# from sqlalchemy.orm import relationship
from .base import Base

class Point(Base):
    __tablename__ = "point"

    id = Column(Integer, primary_key=True, autoincrement=True)
    startpos = Column(Integer)
    endpos = Column(Integer)

    # sentence_id = Column(String, ForeignKey("sentence.id_sen"))
    
    # sentence = relationship("Sentence", back_populates="points")

from sqlalchemy import Column, String, Text
# from sqlalchemy.orm import relationship
from .base import Base

class Sentence(Base):
    __tablename__ = "sentence"

    id_sen = Column(String, primary_key=True)
    left = Column(Text)
    center = Column(Text)
    right = Column(Text)

    # points = relationship("Point", back_populates="sentence")
    # words = relationship("RowWord", back_populates="sentence")

from sqlalchemy.orm import Session
from schemas import RowWordCreate
from models import RowWord


def create_row_word(db: Session, word: RowWordCreate):
    db_word = RowWord(**word.dict())
    db.add(db_word)
    db.commit()
    db.refresh(db_word)
    return db_word

def get_all_row_words(db: Session):
    return db.query(RowWord).all()
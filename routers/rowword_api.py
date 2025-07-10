from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
from schemas import RowWordCreate, RowWordRead
from crud import create_row_word, get_all_row_words

router = APIRouter()

@router.post("/words/", response_model=RowWordRead)
def create(word: RowWordCreate, db: Session = Depends(get_db)):
    return create_row_word(db, word)

@router.get("/words/", response_model=list[RowWordRead])
def get_all(db: Session = Depends(get_db)):
    return get_all_row_words(db)


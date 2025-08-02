from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from database import get_db
from schemas import RowWordCreate, RowWordRead, RowWordUpdate
from crud import create_row_word, get_all_row_words
from models import RowWord, Corpus
# import pandas as pd
import io
from typing import List, Optional
from pydantic import BaseModel

router = APIRouter()

@router.post("/words/", response_model=RowWordRead)
def create(word: RowWordCreate, db: Session = Depends(get_db)):
    return create_row_word(db, word)

@router.get("/words/")
def get_all(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None),
    pos_filter: Optional[str] = Query(None),
    ner_filter: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get all row words with pagination and filtering"""
    try:
        query = db.query(RowWord)
        
        # Apply search filter
        if search:
            query = query.filter(
                RowWord.Word.contains(search) |
                RowWord.Lemma.contains(search) |
                RowWord.Semantic.contains(search)
            )
        
        # Apply POS filter
        if pos_filter:
            query = query.filter(RowWord.POS == pos_filter)
        
        # Apply NER filter
        if ner_filter:
            query = query.filter(RowWord.NER == ner_filter)
        
        # Apply pagination
        total = query.count()
        words = query.offset(skip).limit(limit).all()
        
        # Convert to dict for JSON serialization
        words_dict = []
        for word in words:
            words_dict.append({
                "ID": word.ID,
                "ID_sen": word.ID_sen,
                "Word": word.Word,
                "Lemma": word.Lemma,
                "Links": word.Links,
                "Morph": word.Morph,
                "POS": word.POS,
                "Phrase": word.Phrase,
                "Grm": word.Grm,
                "NER": word.NER,
                "Semantic": word.Semantic
            })
        
        return words_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/words/{word_id}", response_model=RowWordRead)
def get_word(word_id: str, db: Session = Depends(get_db)):
    """Get a specific word by ID"""
    word = db.query(RowWord).filter(RowWord.ID == word_id).first()
    if not word:
        raise HTTPException(status_code=404, detail="Word not found")
    return word

@router.get("/words/stats/")
def get_word_stats(db: Session = Depends(get_db)):
    """Get statistics about the corpus"""
    total_words = db.query(RowWord).count()
    unique_words = db.query(RowWord.Word).distinct().count()
    unique_pos = db.query(RowWord.POS).distinct().count()
    unique_ner = db.query(RowWord.NER).distinct().count()
    
    # Get POS distribution
    pos_distribution = db.query(RowWord.POS, db.func.count(RowWord.ID)).group_by(RowWord.POS).all()
    
    # Get NER distribution
    ner_distribution = db.query(RowWord.NER, db.func.count(RowWord.ID)).group_by(RowWord.NER).all()
    
    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "unique_pos": unique_pos,
        "unique_ner": unique_ner,
        "pos_distribution": dict(pos_distribution),
        "ner_distribution": dict(ner_distribution)
    }

@router.get("/words/search/")
def search_words(
    q: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """Search words by text"""
    query = db.query(RowWord).filter(
        RowWord.Word.contains(q) |
        RowWord.Lemma.contains(q) |
        RowWord.Semantic.contains(q)
    ).limit(limit)
    
    words = query.all()
    return {"data": words, "query": q, "count": len(words)}

@router.post("/import-rowwords/")
async def import_row_words(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        content = await file.read()
        lines = content.decode('utf-8').splitlines()
        
        count = 0
        for line in lines:
            fields = line.strip().split('\t')
            if len(fields) != 10:
                continue  # skip malformed lines

            row = RowWord(
                ID=fields[0],
                Word=fields[1],
                Lemma=fields[2],
                ID_sen=fields[3],
                Links=fields[4],
                POS=fields[5],
                Phrase=fields[6],
                Grm=fields[7],
                NER=fields[8],
                Semantic=fields[9],
            )
            db.merge(row)
            count += 1

        db.commit()
        return {"message": f"✅ Imported {count} rows successfully."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")

@router.post("/import-corpus-file/")
async def import_corpus_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        # Đọc file content
        content = await file.read()
        filename = file.filename.lower()

        # Đọc file bằng Pandas
        # Temporarily disabled pandas import
        # if filename.endswith(".csv"):
        #     df = pd.read_csv(io.StringIO(content.decode("utf-8")), sep=",")
        # elif filename.endswith(".xlsx"):
        #     df = pd.read_excel(io.BytesIO(content), engine='openpyxl')
        # else:
        raise HTTPException(status_code=501, detail="File import temporarily disabled")

        # Kiểm tra cột cần thiết
        required_columns = ["ID", "ID_sen", "Word", "Lemma", "Links", "Morph", "POS", "Phrase", "Grm", "NER", "Semantic"]
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing columns: {', '.join(missing)}")

        # Thêm vào DB
        count = 0
        for _, row in df.iterrows():
            item = RowWord(
                ID=row["ID"],
                ID_sen=row["ID_sen"],
                Word=row["Word"],
                Lemma=row["Lemma"],
                Links=row["Links"],
                Morph=row["Morph"],
                POS=row["POS"],
                Phrase=row["Phrase"],
                Grm=row["Grm"],
                NER=row["NER"],
                Semantic=row["Semantic"],
            )
            db.merge(item)
            count += 1

        db.commit()
        return {"message": f"✅ Imported {count} rows from {filename}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")

@router.get("/export-rowwords-excel/")
def export_row_words_excel(db: Session = Depends(get_db)):
    # Lấy dữ liệu từ database
    rows = db.query(RowWord).all()

    # Convert sang pandas DataFrame
    data = [{
        "ID": r.ID,
        "ID_sen": r.ID_sen,
        "Word": r.Word,
        "Lemma": r.Lemma,
        "Links": r.Links,
        "Morph": r.Morph,
        "POS": r.POS,
        "Phrase": r.Phrase,
        "Grm": r.Grm,
        "NER": r.NER,
        "Semantic": r.Semantic
    } for r in rows]

    df = pd.DataFrame(data)

    # Ghi vào Excel file trong bộ nhớ
    # Temporarily disabled pandas export
    # output = io.BytesIO()
    # with pd.ExcelWriter(output, engine='openpyxl') as writer:
    #     df.to_excel(writer, index=False, sheet_name="RowWords")
    raise HTTPException(status_code=501, detail="Excel export temporarily disabled")

    output.seek(0)

    # Trả về file Excel để tải
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=row_words.xlsx"}
    )

@router.delete("/words/{word_id}")
def delete_word(word_id: str, db: Session = Depends(get_db)):
    """Delete a word by ID"""
    word = db.query(RowWord).filter(RowWord.ID == word_id).first()
    if not word:
        raise HTTPException(status_code=404, detail="Word not found")
    
    db.delete(word)
    db.commit()
    return {"message": "Word deleted successfully"}

@router.put("/words/{word_id}", response_model=RowWordRead)
def update_word(word_id: str, word_update: RowWordUpdate, db: Session = Depends(get_db)):
    """Update a word by ID"""
    word = db.query(RowWord).filter(RowWord.ID == word_id).first()
    if not word:
        raise HTTPException(status_code=404, detail="Word not found")
    
    # Only update fields that are provided (not None)
    update_data = word_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        if value is not None:
            setattr(word, field, value)
    
    db.commit()
    db.refresh(word)
    return word

class BatchWordsRequest(BaseModel):
    words: List[RowWordCreate]

@router.post("/words/batch/", response_model=dict)
def create_batch_words(batch_request: BatchWordsRequest, db: Session = Depends(get_db)):
    """Create multiple words in batch and also create sentences in corpus table"""
    try:
        success_count = 0
        error_count = 0
        errors = []
        
        # Group words by sentence ID
        sentences = {}
        for word_data in batch_request.words:
            sentence_id = word_data.ID_sen
            if sentence_id not in sentences:
                sentences[sentence_id] = []
            sentences[sentence_id].append(word_data)
        
        # Process words
        for word_data in batch_request.words:
            try:
                # Check if word already exists
                existing_word = db.query(RowWord).filter(RowWord.ID == word_data.ID).first()
                if existing_word:
                    # Update existing word
                    for field, value in word_data.dict().items():
                        setattr(existing_word, field, value)
                    success_count += 1
                else:
                    # Create new word
                    new_word = RowWord(**word_data.dict())
                    db.add(new_word)
                    success_count += 1
            except Exception as e:
                error_count += 1
                errors.append(f"Error with word {word_data.ID}: {str(e)}")
        
        # Process sentences for corpus table
        corpus_success_count = 0
        for sentence_id, words in sentences.items():
            try:
                # Check if sentence already exists in corpus
                existing_sentence = db.query(Corpus).filter(Corpus.sentence_id == sentence_id).first()
                
                if not existing_sentence:
                    # Create sentence text from words
                    sentence_text = " ".join([word.Word for word in words if word.Word])
                    
                    # Determine language from sentence ID (ED = English, VD = Vietnamese)
                    language = "en" if sentence_id.startswith("ED") else "vi" if sentence_id.startswith("VD") else "en"
                    
                    # Create new sentence in corpus
                    new_sentence = Corpus(
                        sentence_id=sentence_id,
                        language=language,
                        text=sentence_text,
                        translation=None,  # Will be filled later if needed
                        meta_data=f"Created from {len(words)} words"
                    )
                    db.add(new_sentence)
                    corpus_success_count += 1
            except Exception as e:
                errors.append(f"Error with sentence {sentence_id}: {str(e)}")
        
        db.commit()
        
        return {
            "success": True,
            "message": f"Batch operation completed. {success_count} words and {corpus_success_count} sentences processed successfully, {error_count} failed.",
            "success_count": success_count,
            "corpus_success_count": corpus_success_count,
            "error_count": error_count,
            "errors": errors if errors else None
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Batch operation failed: {str(e)}")

from fastapi import APIRouter, UploadFile, File, Depends, Form, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from database import get_db
from schemas import RowWordCreate, RowWordRead
from crud import create_row_word, get_all_row_words
from models import RowWord, Sentence, Point
import pandas as pd
import io
from collections import defaultdict

router = APIRouter()

@router.post("/words/", response_model=RowWordRead)
def create(word: RowWordCreate, db: Session = Depends(get_db)):
    return create_row_word(db, word)

@router.get("/words/")
def get_all(db: Session = Depends(get_db),lang_code: str = '', search: str = '', page: int = 1, limit: int = 10):
    query = db.query(RowWord)

    if lang_code != '':
        query = query.filter(RowWord.Lang_code == lang_code)

    if search != '':
        query = query.filter(RowWord.Word.contains(search))

    total = query.count()
    total_pages = (total + limit - 1) // limit

    data = query.offset((page - 1) * limit).limit(limit).all()

    return {"data": data, "page": page, "limit": limit, "total": total, "total_pages": total_pages}

    # return get_all_row_words(db)

@router.post("/import-rowwords/")
async def import_row_words(
    file: UploadFile = File(...),
    lang_code: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        content = await file.read()
        lines = content.decode('utf-8').splitlines()

        rowwords_by_sentence = defaultdict(list)
        count = 0

        for line in lines:
            fields = line.strip().split('\t')
            if len(fields) != 10:
                continue  # skip malformed lines
            ID = fields[0]
            ID_sen = ID[2:-2]
            row = RowWord(
                ID=fields[0],
                Word=fields[1],
                Lemma=fields[2],
                ID_sen=ID_sen,
                Links=fields[4],
                POS=fields[5],
                Phrase=fields[6],
                Grm=fields[7],
                NER=fields[8],
                Semantic=fields[9],
                Lang_code=lang_code
            )
            rowwords_by_sentence[row.ID_sen].append(row)

            db.merge(row)  # upsert RowWord
            count += 1

        # Tạo Sentence + Point sau khi đã gom nhóm theo câu
        # for id_sen, rowwords in rowwords_by_sentence.items():
        #     # Tạo Sentence nếu chưa có
        #     sentence_exists = db.query(Sentence).filter_by(id_sen=id_sen).first()
        #     if not sentence_exists:
        #         sentence = Sentence(
        #             id_sen=id_sen,
        #             left="",
        #             center=" ".join([rw.Word for rw in rowwords]),
        #             right=""
        #         )
        #         db.add(sentence)

            # Tạo Point (startpos=0, endpos=N-1)
            # point_exists = db.query(Point).filter_by(sentence_id=id_sen).first()
            # if not point_exists:
            #     point = Point(
            #         sentence_id=id_sen,
            #         startpos=0,
            #         endpos=len(rowwords) - 1
            #     )
            #     db.add(point)

        db.commit()
        return {
            "message": f"Imported {count} row words from file.",
            "sentences": len(rowwords_by_sentence)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@router.get("/sentences")
async def get_sentences(db: Session = Depends(get_db)):
    rows = db.query(RowWord).order_by(RowWord.ID_sen).all()
    rows_dict = [{"ID_sen": r.ID_sen} for r in rows]

    def build_sentence_map(rows):
        dicIdEng = {}
        if not rows:
            return dicIdEng

        idSen = rows[0]['ID_sen']
        start = 0

        for i in range(len(rows)):
            current_id = rows[i]['ID_sen']

            if current_id != idSen:
                dicIdEng[idSen] = {"start": start, "end": i - 1}
                idSen = current_id
                start = i
            elif i == len(rows) - 1:
                dicIdEng[idSen] = {"start": start, "end": i}

        return dicIdEng

    result = build_sentence_map(rows_dict)
    return result

@router.post("/import-corpus-file/")
async def import_corpus_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        # Đọc file content
        content = await file.read()
        filename = file.filename.lower()

        # Đọc file bằng Pandas
        if filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(content.decode("utf-8")), sep=",")
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content), engine='openpyxl')
        else:
            raise HTTPException(status_code=400, detail="File must be .csv or .xlsx")

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
        return {"message": f" Imported {count} rows from {filename}"}

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
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="RowWords")

    output.seek(0)

    # Trả về file Excel để tải
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=row_words.xlsx"}
    )

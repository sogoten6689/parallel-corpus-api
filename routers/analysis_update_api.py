from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from database import get_db
from models import RowWord
from pydantic import BaseModel

router = APIRouter()

class AnalysisRowUpdate(BaseModel):
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

class AnalysisRowsUpdate(BaseModel):
    rows: List[AnalysisRowUpdate]

class AnalysisDataUpdate(BaseModel):
    Word: str
    Lemma: str
    Links: str
    Morph: str
    POS: str
    Phrase: str
    Grm: str
    NER: str
    Semantic: str

class AnalysisDataRowsUpdate(BaseModel):
    rows: List[AnalysisDataUpdate]

@router.put("/update-row/")
async def update_analysis_row(row_data: AnalysisRowUpdate, db: Session = Depends(get_db)):
    """Update a single analysis row"""
    try:
        # Find existing row by ID
        existing_row = db.query(RowWord).filter(RowWord.ID == row_data.ID).first()
        
        if not existing_row:
            # Create new row if doesn't exist
            new_row = RowWord(**row_data.dict())
            db.add(new_row)
            db.commit()
            db.refresh(new_row)
            return {"success": True, "message": "Row created successfully", "data": new_row}
        
        # Update existing row
        for field, value in row_data.dict().items():
            setattr(existing_row, field, value)
        
        db.commit()
        db.refresh(existing_row)
        
        return {"success": True, "message": "Row updated successfully", "data": existing_row}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update row: {str(e)}")

@router.put("/update-analysis-row/")
async def update_analysis_row_data(row_data: AnalysisDataUpdate, db: Session = Depends(get_db)):
    """Update a single analysis row without ID and ID_sen"""
    try:
        # Try to find existing row by Word and other fields
        existing_row = db.query(RowWord).filter(
            RowWord.Word == row_data.Word,
            RowWord.Lemma == row_data.Lemma,
            RowWord.POS == row_data.POS
        ).first()
        
        if not existing_row:
            # Create new row with generated ID
            import uuid
            generated_id = f"analysis_{uuid.uuid4().hex[:8]}"
            generated_id_sen = f"sentence_{uuid.uuid4().hex[:4]}"
            
            new_row_data = {
                "ID": generated_id,
                "ID_sen": generated_id_sen,
                **row_data.dict()
            }
            new_row = RowWord(**new_row_data)
            db.add(new_row)
            db.commit()
            db.refresh(new_row)
            return {"success": True, "message": "Row created successfully", "data": new_row}
        
        # Update existing row
        for field, value in row_data.dict().items():
            setattr(existing_row, field, value)
        
        db.commit()
        db.refresh(existing_row)
        
        return {"success": True, "message": "Row updated successfully", "data": existing_row}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update analysis row: {str(e)}")

@router.put("/update-rows/")
async def update_analysis_rows(rows_data: AnalysisRowsUpdate, db: Session = Depends(get_db)):
    """Update multiple analysis rows"""
    try:
        updated_count = 0
        created_count = 0
        
        for row_data in rows_data.rows:
            # Find existing row by ID
            existing_row = db.query(RowWord).filter(RowWord.ID == row_data.ID).first()
            
            if not existing_row:
                # Create new row if doesn't exist
                new_row = RowWord(**row_data.dict())
                db.add(new_row)
                created_count += 1
            else:
                # Update existing row
                for field, value in row_data.dict().items():
                    setattr(existing_row, field, value)
                updated_count += 1
        
        db.commit()
        
        return {
            "success": True, 
            "message": f"Updated {updated_count} rows, created {created_count} rows",
            "data": {"updated": updated_count, "created": created_count}
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update rows: {str(e)}")

@router.put("/update-analysis-data/")
async def update_analysis_data(rows_data: AnalysisDataRowsUpdate, db: Session = Depends(get_db)):
    """Update analysis data without ID and ID_sen (for frontend analysis table)"""
    try:
        updated_count = 0
        created_count = 0
        
        for i, row_data in enumerate(rows_data.rows):
            # Generate ID and ID_sen for new rows
            generated_id = f"analysis_{i}_{row_data.Word}"
            generated_id_sen = f"sentence_{i}"
            
            # Try to find existing row by Word and other fields
            existing_row = db.query(RowWord).filter(
                RowWord.Word == row_data.Word,
                RowWord.Lemma == row_data.Lemma,
                RowWord.POS == row_data.POS
            ).first()
            
            if not existing_row:
                # Create new row with generated ID
                new_row_data = {
                    "ID": generated_id,
                    "ID_sen": generated_id_sen,
                    **row_data.dict()
                }
                new_row = RowWord(**new_row_data)
                db.add(new_row)
                created_count += 1
            else:
                # Update existing row
                for field, value in row_data.dict().items():
                    setattr(existing_row, field, value)
                updated_count += 1
        
        db.commit()
        
        return {
            "success": True, 
            "message": f"Updated {updated_count} rows, created {created_count} rows",
            "data": {"updated": updated_count, "created": created_count}
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update analysis data: {str(e)}") 
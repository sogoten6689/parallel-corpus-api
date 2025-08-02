from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import subprocess
import json
import tempfile
import os
from typing import Optional
import pandas as pd
import io
import subprocess
import sys
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import spacy

router = APIRouter()

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

class AutoFillRequest(BaseModel):
    words: List[dict]

class AutoFillResponse(BaseModel):
    success: bool
    message: str
    updated_words: List[dict]

@router.post("/auto-fill-analysis/", response_model=AutoFillResponse)
async def auto_fill_analysis(request: AutoFillRequest):
    """Auto-fill NER, Phrase, and Semantic using spaCy"""
    if nlp is None:
        raise HTTPException(status_code=500, detail="spaCy model not loaded")
    
    try:
        updated_words = []
        
        for word_data in request.words:
            word = word_data.get("Word", "")
            if not word:
                updated_words.append(word_data)
                continue
            
            # Process with spaCy
            doc = nlp(word)
            
            # Get NER
            ner = "O"
            if doc.ents:
                ner = doc.ents[0].label_
            
            # Get phrase/chunk
            phrase = ""
            if len(doc) > 0:
                token = doc[0]
                if token.dep_ in ["nsubj", "dobj", "iobj", "pobj"]:
                    phrase = token.dep_.upper()
                elif token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"]:
                    phrase = token.pos_
                else:
                    phrase = token.dep_.upper()
            
            # Get semantic (simplified)
            semantic = ""
            if doc.ents:
                semantic = doc.ents[0].label_
            elif token.pos_ == "NOUN":
                semantic = "ENTITY"
            elif token.pos_ == "VERB":
                semantic = "ACTION"
            elif token.pos_ == "ADJ":
                semantic = "PROPERTY"
            else:
                semantic = "OTHER"
            
            # Update word data
            updated_word = word_data.copy()
            updated_word["NER"] = ner
            updated_word["Phrase"] = phrase
            updated_word["Semantic"] = semantic
            
            updated_words.append(updated_word)
        
        return AutoFillResponse(
            success=True,
            message=f"Successfully processed {len(updated_words)} words",
            updated_words=updated_words
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-fill failed: {str(e)}")

@router.post("/analyze-text/")
async def analyze_text(
    text: str = Form(...),
    language: str = Form("vi"),
    tab: str = Form("analysis")
):
    """Analyze text using Python scripts"""
    try:
        # Create temporary file for text
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text)
            temp_file = f.name

        # Run analysis script
        script_path = os.path.join(os.getcwd(), "scripts", "analyze_text.py")
        cmd = [
            "python", script_path, 
            text, 
            "--language", language,
            "--tab", tab
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Clean up temp file
        os.unlink(temp_file)
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500, 
                detail=f"Analysis failed: {result.stderr}"
            )
        
        # Parse JSON response
        try:
            analysis_result = json.loads(result.stdout)
            return JSONResponse(content=analysis_result)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail="Invalid JSON response from analysis script"
            )
            
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408,
            detail="Analysis timeout"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis error: {str(e)}"
        )

@router.post("/analyze-file/")
async def analyze_file(file: UploadFile = File(...)):
    """Analyze uploaded file"""
    try:
        # Read file content
        content = await file.read()
        filename = file.filename.lower()
        
        # Determine file type and read with pandas
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(content))
        elif filename.endswith('.txt'):
            # For text files, read as text
            text_content = content.decode('utf-8')
            return await analyze_text(text=text_content)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Use CSV, Excel, or TXT files."
            )
        
        # Process the dataframe
        data_info = {
            "fileName": filename,
            "rows": len(df),
            "columns": len(df.columns),
            "preview": df.head(10).to_dict('records'),
            "columnNames": df.columns.tolist()
        }
        
        return {
            "success": True,
            "dataInfo": data_info
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File analysis failed: {str(e)}"
        )

@router.post("/ai-chat/")
async def ai_chat(
    question: str = Form(...),
    data: Optional[str] = Form(None),
    chat_history: Optional[str] = Form("[]")
):
    """AI chat endpoint"""
    try:
        # Create input data
        input_data = {
            "question": question,
            "data": data,
            "chatHistory": json.loads(chat_history) if chat_history else []
        }
        
        # Run AI chat script
        script_path = os.path.join(os.getcwd(), "scripts", "ai_chat_hf.py")
        
        result = subprocess.run(
            ["python", script_path],
            input=json.dumps(input_data),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"AI chat failed: {result.stderr}"
            )
        
        # Parse response
        try:
            response = json.loads(result.stdout)
            return {
                "success": True,
                "answer": response.get("answer", "No response generated")
            }
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail="Invalid response from AI chat"
            )
            
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408,
            detail="AI chat timeout"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI chat error: {str(e)}"
        )

@router.post("/data-analysis/")
async def data_analysis(
    message: str = Form(...),
    data_info: Optional[str] = Form(None)
):
    """Data analysis endpoint"""
    try:
        # Parse data info
        data = json.loads(data_info) if data_info else {}
        
        # Run XAI analysis script
        script_path = os.path.join(os.getcwd(), "scripts", "xai_analysis.py")
        
        input_data = {
            "message": message,
            "data": data
        }
        
        result = subprocess.run(
            ["python", script_path],
            input=json.dumps(input_data),
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Data analysis failed: {result.stderr}"
            )
        
        # Parse response
        try:
            response = json.loads(result.stdout)
            return {
                "success": True,
                "result": response.get("result", ""),
                "xai": response.get("xai", [])
            }
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail="Invalid response from data analysis"
            )
            
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408,
            detail="Data analysis timeout"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data analysis error: {str(e)}"
        )

@router.post("/semantic-alignment/")
async def semantic_alignment(
    text1: str = Form(...),
    text2: str = Form(...)
):
    """Semantic alignment endpoint"""
    try:
        # Run semantic alignment script
        script_path = os.path.join(os.getcwd(), "scripts", "semantic_alignment_v2.py")
        
        input_data = {
            "text1": text1,
            "text2": text2
        }
        
        result = subprocess.run(
            ["python", script_path],
            input=json.dumps(input_data),
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Semantic alignment failed: {result.stderr}"
            )
        
        # Parse response
        try:
            response = json.loads(result.stdout)
            return {
                "success": True,
                "alignment": response
            }
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail="Invalid response from semantic alignment"
            )
            
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=408,
            detail="Semantic alignment timeout"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Semantic alignment error: {str(e)}"
        ) 
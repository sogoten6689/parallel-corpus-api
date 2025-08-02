#!/usr/bin/env python3
"""
Advanced Phrase Chunking using modern NLP models
Supports multiple approaches: spaCy, transformers, and rule-based
"""

import sys
import json
import warnings
warnings.filterwarnings("ignore")

def chunk_with_spacy(text, language='en'):
    """Phrase chunking using spaCy's built-in phrase detection"""
    try:
        import spacy
        
        if language == 'en':
            nlp = spacy.load("en_core_web_sm")
        else:
            # For Vietnamese, use basic tokenization
            return chunk_with_rules(text, language)
            
        doc = nlp(text)
        
        chunks = []
        
        # Noun phrases
        for chunk in doc.noun_chunks:
            chunks.append({
                "text": chunk.text,
                "start": chunk.start,
                "end": chunk.end,
                "type": "NP",
                "label": "Noun Phrase",
                "tokens": [token.text for token in chunk]
            })
        
        # Verb phrases (custom detection)
        verb_phrases = []
        for token in doc:
            if token.pos_ == "VERB":
                # Find verb phrase boundaries
                start_idx = token.i
                end_idx = token.i + 1
                
                # Extend to include auxiliaries before
                for i in range(token.i - 1, -1, -1):
                    if doc[i].dep_ in ["aux", "auxpass"] or doc[i].pos_ in ["ADP", "PART"]:
                        start_idx = i
                    else:
                        break
                
                # Extend to include complements after
                for i in range(token.i + 1, len(doc)):
                    if doc[i].dep_ in ["dobj", "prep", "prt", "compound"] or doc[i].pos_ in ["ADP", "PART"]:
                        end_idx = i + 1
                    else:
                        break
                
                if end_idx > start_idx + 1:  # Multi-word phrase
                    phrase_text = " ".join([doc[i].text for i in range(start_idx, end_idx)])
                    verb_phrases.append({
                        "text": phrase_text,
                        "start": start_idx,
                        "end": end_idx,
                        "type": "VP",
                        "label": "Verb Phrase",
                        "tokens": [doc[i].text for i in range(start_idx, end_idx)]
                    })
        
        chunks.extend(verb_phrases)
        
        # Prepositional phrases
        for token in doc:
            if token.pos_ == "ADP":  # Preposition
                start_idx = token.i
                end_idx = token.i + 1
                
                # Extend to include the object
                for i in range(token.i + 1, len(doc)):
                    if doc[i].dep_ in ["pobj", "pcomp"] or doc[i].pos_ in ["NOUN", "PROPN", "PRON"]:
                        end_idx = i + 1
                        # Include any modifiers
                        for j in range(i + 1, len(doc)):
                            if doc[j].dep_ in ["amod", "compound"] and doc[j].head.i == i:
                                end_idx = j + 1
                            else:
                                break
                        break
                    elif doc[i].pos_ in ["DET", "ADJ"]:
                        end_idx = i + 1
                    else:
                        break
                
                if end_idx > start_idx + 1:
                    phrase_text = " ".join([doc[i].text for i in range(start_idx, end_idx)])
                    chunks.append({
                        "text": phrase_text,
                        "start": start_idx,
                        "end": end_idx,
                        "type": "PP",
                        "label": "Prepositional Phrase",
                        "tokens": [doc[i].text for i in range(start_idx, end_idx)]
                    })
        
        return sorted(chunks, key=lambda x: x['start'])
        
    except Exception as e:
        print(f"spaCy chunking error: {e}", file=sys.stderr)
        return chunk_with_rules(text, language)

def chunk_with_transformers(text, language='en'):
    """Phrase chunking using BERT-based models"""
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
        
        # Use a pre-trained chunking model
        if language == 'en':
            # Use GLiNER for general NER/chunking
            try:
                from gliner import GLiNER
                model = GLiNER.from_pretrained("urchade/gliner_small-v1")
                
                # Define phrase types to detect
                labels = ["noun phrase", "verb phrase", "prepositional phrase", "time phrase", "location phrase"]
                
                entities = model.predict_entities(text, labels, threshold=0.3)
                
                chunks = []
                for i, entity in enumerate(entities):
                    chunks.append({
                        "text": entity["text"],
                        "start": entity["start"],
                        "end": entity["end"],
                        "type": entity["label"].upper().replace(" ", "_"),
                        "label": entity["label"].title(),
                        "tokens": entity["text"].split(),
                        "confidence": entity.get("score", 0.5)
                    })
                
                return sorted(chunks, key=lambda x: x['start'])
                
            except ImportError:
                # Fallback to BERT-based NER
                ner_pipeline = pipeline("ner", 
                                      model="dslim/bert-large-NER",
                                      aggregation_strategy="simple")
                
                entities = ner_pipeline(text)
                
                chunks = []
                for entity in entities:
                    chunks.append({
                        "text": entity["word"],
                        "start": entity["start"],
                        "end": entity["end"],
                        "type": "NP",
                        "label": f"Named Entity ({entity['entity_group']})",
                        "tokens": entity["word"].split(),
                        "confidence": entity["score"]
                    })
                
                return chunks
        
        else:
            # For Vietnamese, use rule-based approach
            return chunk_with_rules(text, language)
            
    except Exception as e:
        print(f"Transformers chunking error: {e}", file=sys.stderr)
        return chunk_with_rules(text, language)

def get_dynamic_patterns(words, language='vi'):
    """Dynamically generate phrase patterns based on input text"""
    if language == 'vi':
        return get_vietnamese_patterns(words)
    else:
        return get_english_patterns(words)

def get_vietnamese_patterns(words):
    """Generate Vietnamese patterns based on POS and semantic roles"""
    patterns = {
        "subject": [],
        "verb": [], 
        "location": [],
        "time": [],
        "object": [],
        "quantity": [],
        "adjective": []
    }
    
    # Base patterns (expandable)
    base_patterns = {
        # Pronouns and subjects
        "subject": [
            ["tôi"], ["chúng", "tôi"], ["chúng", "ta"], ["chúng", "nó"],
            ["anh"], ["chị"], ["em"], ["bạn"], ["người"], 
            ["anh", "ấy"], ["chị", "ấy"], ["em", "ấy"],
            ["ông"], ["bà"], ["cô"], ["chú"], ["thầy"], ["cô", "giáo"]
        ],
        # Verbs with aspects/tenses
        "verb": [
            ["đã"], ["đang"], ["sẽ"], ["có", "thể"], ["cần"], ["muốn"],
            ["học"], ["đi"], ["làm"], ["ăn"], ["uống"], ["ngủ"], ["chơi"],
            ["đọc"], ["viết"], ["nói"], ["nghe"], ["xem"], ["mua"], ["bán"]
        ],
        # Location indicators
        "location": [
            ["ở"], ["trong"], ["tại"], ["trên"], ["dưới"], ["bên"],
            ["trường"], ["nhà"], ["công", "ty"], ["văn", "phòng"],
            ["phòng"], ["lớp"], ["sân"], ["vườn"], ["đường"]
        ],
        # Time expressions
        "time": [
            ["hôm", "nay"], ["ngày", "mai"], ["hôm", "qua"],
            ["buổi", "sáng"], ["buổi", "chiều"], ["buổi", "tối"],
            ["cả", "ngày"], ["suốt", "ngày"], ["cả", "đêm"],
            ["lúc"], ["khi"], ["vào"], ["từ"], ["đến"]
        ],
        # Objects and things
        "object": [
            ["bài"], ["cuốn"], ["quyển"], ["chiếc"], ["cái"], ["con"],
            ["sách"], ["vở"], ["bút"], ["máy"], ["xe"], ["nhà"]
        ],
        # Quantities and numbers
        "quantity": [
            ["một"], ["hai"], ["ba"], ["nhiều"], ["ít"], ["vài"],
            ["tất", "cả"], ["toàn", "bộ"], ["một", "số"]
        ],
        # Adjectives and descriptors
        "adjective": [
            ["tốt"], ["xấu"], ["đẹp"], ["to"], ["nhỏ"], ["cao"], ["thấp"],
            ["nhanh"], ["chậm"], ["mới"], ["cũ"], ["trẻ"], ["già"]
        ]
    }
    
    # Expand patterns by finding combinations in the text
    text_lower = [w.lower() for w in words]
    
    for category, base_list in base_patterns.items():
        # Add base patterns
        patterns[category].extend(base_list)
        
        # Find dynamic combinations
        for i in range(len(text_lower)):
            for j in range(i + 1, min(i + 4, len(text_lower))):  # Max 4 words
                phrase = text_lower[i:j]
                
                # Check if phrase contains category markers
                if category == "verb" and any(w in phrase for w in ["đã", "đang", "sẽ", "có", "thể"]):
                    if phrase not in patterns[category]:
                        patterns[category].append(phrase)
                
                elif category == "location" and any(w in phrase for w in ["ở", "trong", "tại", "trên", "dưới"]):
                    if phrase not in patterns[category]:
                        patterns[category].append(phrase)
                
                elif category == "time" and any(w in phrase for w in ["buổi", "cả", "suốt", "vào", "lúc"]):
                    if phrase not in patterns[category]:
                        patterns[category].append(phrase)
    
    return patterns

def get_english_patterns(words):
    """Generate English patterns using basic POS heuristics"""
    patterns = {
        "subject": [],
        "verb": [],
        "location": [],
        "time": [],
        "object": [],
        "quantity": [],
        "adjective": []
    }
    
    # Base English patterns
    base_patterns = {
        "subject": [
            ["i"], ["you"], ["he"], ["she"], ["it"], ["we"], ["they"],
            ["this"], ["that"], ["these"], ["those"], ["the"], ["a"], ["an"]
        ],
        "verb": [
            ["am"], ["is"], ["are"], ["was"], ["were"], ["have"], ["has"], ["had"],
            ["will"], ["would"], ["can"], ["could"], ["should"], ["may"], ["might"],
            ["go"], ["come"], ["see"], ["do"], ["make"], ["get"], ["take"], ["give"]
        ],
        "location": [
            ["in"], ["on"], ["at"], ["under"], ["over"], ["near"], ["by"],
            ["school"], ["home"], ["office"], ["park"], ["street"], ["room"]
        ],
        "time": [
            ["today"], ["tomorrow"], ["yesterday"], ["now"], ["then"],
            ["morning"], ["afternoon"], ["evening"], ["night"],
            ["all", "day"], ["all", "night"], ["at"], ["when"], ["while"]
        ],
        "object": [
            ["book"], ["pen"], ["car"], ["house"], ["food"], ["water"],
            ["lesson"], ["homework"], ["exercise"], ["test"]
        ],
        "quantity": [
            ["one"], ["two"], ["three"], ["many"], ["few"], ["some"],
            ["all"], ["every"], ["each"], ["several"]
        ],
        "adjective": [
            ["good"], ["bad"], ["big"], ["small"], ["fast"], ["slow"],
            ["new"], ["old"], ["young"], ["beautiful"], ["nice"]
        ]
    }
    
    return base_patterns

def chunk_with_rules(text, language='vi'):
    """Enhanced rule-based phrase chunking with dynamic patterns"""
    words = text.split()
    chunks = []
    
    # Get dynamic patterns based on the input
    patterns = get_dynamic_patterns(words, language)
    
    used_positions = set()
    
    # Find pattern matches
    for phrase_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            pattern_len = len(pattern)
            
            for i in range(len(words) - pattern_len + 1):
                if any(pos in used_positions for pos in range(i, i + pattern_len)):
                    continue
                    
                word_slice = [w.lower() for w in words[i:i + pattern_len]]
                
                if word_slice == pattern:
                    chunks.append({
                        "text": " ".join(words[i:i + pattern_len]),
                        "start": i,
                        "end": i + pattern_len,
                        "type": phrase_type.upper(),
                        "label": f"Cụm {phrase_type}",
                        "tokens": words[i:i + pattern_len],
                        "confidence": 0.9
                    })
                    
                    # Mark positions as used
                    for pos in range(i, i + pattern_len):
                        used_positions.add(pos)
    
    # Add remaining single words as individual chunks
    for i, word in enumerate(words):
        if i not in used_positions:
            chunks.append({
                "text": word,
                "start": i,
                "end": i + 1,
                "type": "WORD",
                "label": "Từ đơn",
                "tokens": [word],
                "confidence": 0.5
            })
    
    return sorted(chunks, key=lambda x: x['start'])

def main():
    """Main function to handle phrase chunking"""
    if len(sys.argv) < 3:
        sys.exit(1)
    
    text = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "rules"
    language = sys.argv[3] if len(sys.argv) > 3 else "vi"
    
    if not text:
        sys.exit(1)
    
    # Choose chunking method
    if method == "spacy":
        chunks = chunk_with_spacy(text, language)
    elif method == "transformers":
        chunks = chunk_with_transformers(text, language)
    else:
        chunks = chunk_with_rules(text, language)
    
    # Group chunks by type for better alignment
    grouped_chunks = {}
    for chunk in chunks:
        chunk_type = chunk['type']
        if chunk_type not in grouped_chunks:
            grouped_chunks[chunk_type] = []
        grouped_chunks[chunk_type].append(chunk)
    
    result = {
        "chunks": chunks,
        "grouped_chunks": grouped_chunks,
        "total_chunks": len(chunks),
        "method": method,
        "language": language,
        "statistics": {
            chunk_type: len(chunk_list) 
            for chunk_type, chunk_list in grouped_chunks.items()
        }
    }
    
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main() 
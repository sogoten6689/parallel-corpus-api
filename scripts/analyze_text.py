#!/usr/bin/env python3
"""
Script để phân tích ngôn ngữ với dịch hai chiều và tối ưu hiệu năng
"""

import sys
import json
import subprocess
import tempfile
import os
from pathlib import Path

# Thêm thư mục gốc vào Python path
sys.path.append(str(Path(__file__).parent.parent))

def load_models():
    """Tải các model cần thiết"""
    try:
        import spacy
        from transformers import pipeline
        from simalign import SentenceAligner
        
        # Suppress print statements for clean JSON output
        import sys
        import os
        sys.stdout = open(os.devnull, 'w')
        
        nlp = spacy.load("en_core_web_sm")
        translator_vi_to_en = pipeline("translation_vi_to_en", model="Helsinki-NLP/opus-mt-vi-en")
        translator_en_to_vi = pipeline("translation_en_to_vi", model="Helsinki-NLP/opus-mt-en-vi")
        sim_aligner = SentenceAligner(model="bert", token_type="word", matching_methods="i")
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        return nlp, translator_vi_to_en, translator_en_to_vi, sim_aligner
    except Exception as e:
        # Restore stdout in case of error
        sys.stdout = sys.__stdout__
        return None, None, None, None

def translate_vi_to_en(text):
    """Dịch từ tiếng Việt sang tiếng Anh"""
    if not text:
        return ""
    try:
        from transformers import pipeline
        translator = pipeline("translation_vi_to_en", model="Helsinki-NLP/opus-mt-vi-en")
        return translator(text)[0]['translation_text']
    except Exception as e:
        return ""

def translate_en_to_vi(text):
    """Dịch từ tiếng Anh sang tiếng Việt"""
    if not text:
        return ""
    try:
        from transformers import pipeline
        translator = pipeline("translation_en_to_vi", model="Helsinki-NLP/opus-mt-en-vi")
        return translator(text)[0]['translation_text']
    except Exception as e:
        return ""

def get_mock_vietnamese_analysis(text):
    """Tạo phân tích tiếng Việt với logic cải thiện"""
    words = text.split()
    analysis = []
    
    # Vietnamese dictionaries
    pronouns = {"tôi", "chúng", "anh", "chị", "em", "bạn", "mình", "ta", "nó", "họ"}
    verbs = {"học", "đi", "làm", "ăn", "nói", "viết", "đọc", "chơi", "nghỉ", "ngủ", "là", "có", "được", "đã", "sẽ", "đang"}
    nouns = {"trường", "nhà", "bài", "sách", "người", "ngày", "thời", "gian", "lớp", "phòng", "xe", "nước"}
    adjectives = {"tốt", "đẹp", "to", "nhỏ", "cao", "thấp", "nhanh", "chậm", "mới", "cũ", "cả", "toàn"}
    prepositions = {"ở", "trong", "trên", "dưới", "bên", "với", "của", "cho", "về", "từ", "đến"}
    conjunctions = {"và", "hoặc", "nhưng", "mà", "để", "vì", "nên"}
    determiners = {"này", "đó", "kia", "những", "các", "mỗi", "mọi"}
    
    # Find main verb for better dependency structure
    main_verb_idx = -1
    for i, word in enumerate(words):
        if word.lower() in verbs and word.lower() not in {"là", "có", "được", "đã", "sẽ", "đang"}:
            main_verb_idx = i
            break
    if main_verb_idx == -1:  # fallback to auxiliary verbs
        for i, word in enumerate(words):
            if word.lower() in {"là", "có", "được"}:
                main_verb_idx = i
                break
    
    for i, word in enumerate(words):
        word_lower = word.lower()
        
        # Determine POS
        if word_lower in pronouns:
            pos = "PRON"
            phrase = "NP"
        elif word_lower in verbs:
            pos = "VERB"
            phrase = "VP"
        elif word_lower in nouns:
            pos = "NOUN"
            phrase = "NP"
        elif word_lower in adjectives:
            pos = "ADJ"
            phrase = "ADJP"
        elif word_lower in prepositions:
            pos = "ADP"
            phrase = "PP"
        elif word_lower in conjunctions:
            pos = "CCONJ"
            phrase = "N/A"
        elif word_lower in determiners:
            pos = "DET"
            phrase = "NP"
        else:
            # Default classification
            if word[0].isupper():
                pos = "PROPN"
                phrase = "NP"
            else:
                pos = "NOUN"
                phrase = "NP"
        
        # Determine dependency links
        if main_verb_idx != -1 and i == main_verb_idx:
            # Main verb is ROOT
            links = f"{word} -> ROOT"
            grm = "ROOT"
        elif pos == "PRON" and main_verb_idx != -1 and i < main_verb_idx:
            # Pronoun before verb is subject
            links = f"{word} -> {words[main_verb_idx]}"
            grm = "nsubj"
        elif pos in ["NOUN", "PROPN"] and main_verb_idx != -1 and i > main_verb_idx:
            # Noun after verb is object
            links = f"{word} -> {words[main_verb_idx]}"
            grm = "dobj"
        elif pos == "ADJ" and i > 0:
            # Adjective modifies previous noun
            prev_idx = i - 1
            while prev_idx >= 0 and words[prev_idx].lower() not in nouns and words[prev_idx].lower() not in pronouns:
                prev_idx -= 1
            if prev_idx >= 0:
                links = f"{word} -> {words[prev_idx]}"
                grm = "amod"
            else:
                links = f"{word} -> {words[main_verb_idx] if main_verb_idx != -1 else words[0]}"
                grm = "acomp"
        elif pos == "ADP" and i < len(words) - 1:
            # Preposition connects to next word
            links = f"{word} -> {words[i + 1]}"
            grm = "case"
        elif pos in ["NOUN", "PROPN"] and i > 0 and words[i-1].lower() in prepositions:
            # Noun after preposition
            links = f"{word} -> {words[main_verb_idx] if main_verb_idx != -1 else words[0]}"
            grm = "nmod"
        else:
            # Default dependency
            if main_verb_idx != -1:
                links = f"{word} -> {words[main_verb_idx]}"
                grm = "dep"
            else:
                links = f"{word} -> {words[0]}" if i > 0 else f"{word} -> ROOT"
                grm = "ROOT" if i == 0 else "dep"
        
        # Determine morphological features for Vietnamese
        morph_features = []
        
        # Aspect markers
        if word_lower == "đã":
            morph_features.append("Aspect=Perf")  # Perfective aspect
        elif word_lower == "đang":
            morph_features.append("Aspect=Prog")  # Progressive aspect
        elif word_lower == "sẽ":
            morph_features.append("Tense=Fut")   # Future tense
        
        # Pronouns
        if word_lower in pronouns:
            if word_lower in {"tôi", "mình"}:
                morph_features.append("Person=1|Number=Sing")
            elif word_lower == "chúng":
                morph_features.append("Person=1|Number=Plur")
            elif word_lower in {"anh", "chị", "em", "bạn"}:
                morph_features.append("Person=2|Number=Sing")
            elif word_lower in {"ta", "nó"}:
                morph_features.append("Person=3|Number=Sing")
            elif word_lower == "họ":
                morph_features.append("Person=3|Number=Plur")
        
        # Classifiers (measure words)
        classifiers = {"cái", "con", "người", "chiếc", "quyển", "cuốn", "bức", "tờ", "cây", "viên"}
        if word_lower in classifiers:
            morph_features.append("NumType=Cls")  # Classifier
        
        # Reduplication (common in Vietnamese)
        if len(word) >= 4 and word[:len(word)//2] == word[len(word)//2:]:
            morph_features.append("Reduplication=Yes")
        
        # Question particles
        if word_lower in {"không", "chưa", "à", "nhỉ", "phải"}:
            morph_features.append("PartType=Int")  # Interrogative particle
        
        # Negation
        if word_lower in {"không", "chưa", "chẳng"}:
            morph_features.append("Polarity=Neg")
        
        # Pluralizers
        if word_lower in {"những", "các"}:
            morph_features.append("Number=Plur")
        
        morph = "|".join(morph_features) if morph_features else "N/A"
        
        # Calculate semantic score
        semantic_score = 0.3  # base score
        if word_lower in pronouns:
            semantic_score += 0.2
        elif word_lower in verbs:
            semantic_score += 0.4
        elif word_lower in nouns:
            semantic_score += 0.3
        elif word_lower in adjectives:
            semantic_score += 0.2
        else:
            semantic_score += len(word) * 0.05
        
        # Normalize to 0-1 range
        semantic_score = min(0.95, max(0.1, semantic_score))
        
        analysis.append({
            "Word": word,
            "Lemma": word.lower(),
            "Links": links,
            "Morph": morph,
            "POS": pos,
            "Phrase": phrase,
            "Grm": grm,
            "NER": "O",
            "Semantic": f"{semantic_score:.2f}"
        })
    
    return analysis

def analyze_english_text(text):
    """Phân tích văn bản tiếng Anh với spaCy"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        
        doc = nlp(text)
        analysis = []
        for token in doc:
            # Calculate semantic score based on spaCy features
            semantic_score = 0.3  # base score
            
            # Add score based on POS tag importance
            if token.pos_ in ["NOUN", "VERB"]:
                semantic_score += 0.3
            elif token.pos_ in ["ADJ", "ADV"]:
                semantic_score += 0.2
            elif token.pos_ in ["PRON", "DET"]:
                semantic_score += 0.1
            
            # Add score based on named entity
            if token.ent_type_:
                semantic_score += 0.2
            
            # Add score based on dependency relation
            if token.dep_ in ["ROOT", "nsubj", "dobj"]:
                semantic_score += 0.2
            elif token.dep_ in ["amod", "advmod"]:
                semantic_score += 0.1
            
            # Add score based on word frequency (inverse)
            if token.is_stop:
                semantic_score -= 0.1
            else:
                semantic_score += 0.1
                
            # Normalize to 0-1 range
            semantic_score = min(0.95, max(0.1, semantic_score))
            
            analysis.append({
                "Word": token.text,
                "Lemma": token.lemma_,
                "Links": f"{token.text} -> {token.head.text}",
                "Morph": str(token.morph) if token.morph else "N/A",
                "POS": token.pos_,
                "Phrase": "NP" if any(chunk.start <= token.i < chunk.end for chunk in doc.noun_chunks) else "N/A",
                "Grm": token.dep_,
                "NER": token.ent_type_ or "O",
                "Semantic": f"{semantic_score:.2f}"
            })
        
        return analysis
    except Exception as e:
        return get_mock_vietnamese_analysis(text)

def get_statistics(analysis):
    """Tính toán thống kê từ kết quả phân tích"""
    if not analysis:
        return {"posCounts": [], "nerCounts": []}
    
    pos_counts = {}
    ner_counts = {}
    
    for item in analysis:
        pos = item.get("POS", "N/A")
        ner = item.get("NER", "O")
        
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
        ner_counts[ner] = ner_counts.get(ner, 0) + 1
    
    return {
        "posCounts": [{"POS": pos, "Count": count} for pos, count in pos_counts.items()],
        "nerCounts": [{"NER": ner, "Count": count} for ner, count in ner_counts.items()]
    }

def get_phrase_chunks(text, language='vi'):
    """Get phrase chunks using external chunker"""
    try:
        import subprocess
        import os
        
        script_path = os.path.join(os.path.dirname(__file__), 'phrase_chunker.py')
        result = subprocess.run([
            'python3', script_path, text, 'rules', language
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            print(f"Phrase chunker error: {result.stderr}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Phrase chunker exception: {e}", file=sys.stderr)
        return None

def get_semantic_phrase_alignment_bidirectional(src_text, trg_text, src_lang='vi', trg_lang='en'):
    """Tạo alignment dựa trên nghĩa cụm từ với Sentence Transformers - hỗ trợ 2 chiều"""
    try:
        # Call enhanced semantic alignment script
        cmd = [
            'python3', 'scripts/semantic_alignment_v2.py'
        ]
        
        # Prepare input data with language information
        input_data = {
            'source': src_text,
            'target': trg_text,
            'source_lang': src_lang,
            'target_lang': trg_lang
        }
        
        result = subprocess.run(
            cmd, 
            input=json.dumps(input_data),
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        if result.returncode == 0:
            alignment_result = json.loads(result.stdout)
            
            if alignment_result.get('status') == 'success':
                return alignment_result.get('alignments', [])
            else:
                print(f"Semantic alignment error: {alignment_result.get('error', 'Unknown error')}")
        else:
            print(f"Semantic alignment script failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("Semantic alignment timed out")
    except Exception as e:
        print(f"Semantic alignment exception: {e}")
    
    # Fallback to simple alignment
    src_words = [w.strip() for w in src_text.split() if w.strip()]
    trg_words = [w.strip() for w in trg_text.split() if w.strip()]
    return align_simple_phrases(src_words, trg_words)

def align_phrase_chunks(src_chunks, trg_chunks, src_words, trg_words):
    """Advanced alignment using phrase chunks"""
    alignment = []
    used_src = set()
    used_trg = set()
    
    # Semantic similarity mapping for chunk types
    chunk_mappings = {
        'SUBJECT': ['NP', 'NOUN_PHRASE'],
        'VERB': ['VP', 'VERB_PHRASE'],
        'LOCATION': ['PP', 'PREPOSITIONAL_PHRASE', 'LOCATION_PHRASE'],
        'TIME': ['TIME_PHRASE', 'PP'],
        'OBJECT': ['NP', 'NOUN_PHRASE']
    }
    
    # Phase 1: Match chunks by semantic type
    for src_chunk in src_chunks:
        if src_chunk['type'] == 'WORD':
            continue
            
        src_type = src_chunk['type']
        best_match = None
        best_score = 0
        
        for trg_chunk in trg_chunks:
            if trg_chunk['type'] == 'WORD':
                continue
                
            # Check if chunk types are semantically compatible
            compatible = False
            for mapped_type, compatible_types in chunk_mappings.items():
                if (src_type == mapped_type and trg_chunk['type'] in compatible_types) or \
                   (src_type in compatible_types and trg_chunk['type'] == mapped_type):
                    compatible = True
                    break
            
            if compatible:
                # Calculate semantic similarity (simplified)
                score = calculate_chunk_similarity(src_chunk, trg_chunk)
                if score > best_score:
                    best_score = score
                    best_match = trg_chunk
        
        if best_match and best_score > 0.3:
            # Create alignment for the phrase
            alignment.append([src_chunk['start'], best_match['start']])
            
            # Mark positions as used
            for pos in range(src_chunk['start'], src_chunk['end']):
                used_src.add(pos)
            for pos in range(best_match['start'], best_match['end']):
                used_trg.add(pos)
    
    # Phase 2: Handle remaining single words
    for i, word in enumerate(src_words):
        if i not in used_src:
            # Find best matching target word
            best_match_idx = None
            best_score = 0
            
            for j, trg_word in enumerate(trg_words):
                if j not in used_trg:
                    score = calculate_word_similarity(word, trg_word)
                    if score > best_score:
                        best_score = score
                        best_match_idx = j
            
            if best_match_idx is not None and best_score > 0.2:
                alignment.append([i, best_match_idx])
                used_src.add(i)
                used_trg.add(best_match_idx)
    
    return sorted(alignment, key=lambda x: x[0])

def calculate_chunk_similarity(src_chunk, trg_chunk):
    """Calculate semantic similarity between chunks using universal patterns"""
    score = 0.0
    
    # Type compatibility bonus
    if src_chunk['type'] == trg_chunk['type']:
        score += 0.5
    
    # Cross-type semantic compatibility
    semantic_mappings = {
        'SUBJECT': ['NP', 'NOUN_PHRASE', 'SUBJECT'],
        'VERB': ['VP', 'VERB_PHRASE', 'VERB'], 
        'LOCATION': ['PP', 'PREPOSITIONAL_PHRASE', 'LOCATION'],
        'TIME': ['TIME_PHRASE', 'TIME', 'PP'],
        'OBJECT': ['NP', 'NOUN_PHRASE', 'OBJECT'],
        'QUANTITY': ['NUM', 'NUMBER', 'QUANTITY'],
        'ADJECTIVE': ['ADJP', 'ADJ_PHRASE', 'ADJECTIVE']
    }
    
    src_type = src_chunk['type']
    trg_type = trg_chunk['type']
    
    for semantic_type, compatible_types in semantic_mappings.items():
        if (src_type == semantic_type and trg_type in compatible_types) or \
           (src_type in compatible_types and trg_type == semantic_type):
            score += 0.3
            break
    
    # Content similarity using dynamic translation detection
    src_text = src_chunk['text'].lower().strip()
    trg_text = trg_chunk['text'].lower().strip()
    
    # Dynamic translation scoring
    content_score = calculate_dynamic_translation_score(src_text, trg_text)
    score += content_score
    
    return min(score, 1.0)

def calculate_dynamic_translation_score(src_text, trg_text):
    """Calculate translation similarity for any Vietnamese-English pair"""
    # Core translation dictionaries (expandable)
    core_translations = {
        # Pronouns
        'tôi': ['i', 'me'], 'chúng tôi': ['we', 'us'], 'chúng ta': ['we', 'us'],
        'anh': ['you', 'he'], 'chị': ['you', 'she'], 'em': ['you'],
        'họ': ['they', 'them'], 'nó': ['it'],
        
        # Common verbs
        'học': ['study', 'learn'], 'đi': ['go'], 'làm': ['do', 'make', 'work'],
        'ăn': ['eat'], 'uống': ['drink'], 'ngủ': ['sleep'], 'chơi': ['play'],
        'đọc': ['read'], 'viết': ['write'], 'nói': ['speak', 'say', 'talk'],
        'xem': ['watch', 'see'], 'mua': ['buy'], 'bán': ['sell'],
        
        # Aspect markers
        'đã': ['have', 'has', 'had', 'did', 'was', 'were'], 
        'đang': ['is', 'are', 'am', 'being'],
        'sẽ': ['will', 'would', 'going to'],
        
        # Locations
        'trường': ['school'], 'nhà': ['home', 'house'], 'công ty': ['company', 'office'],
        'phòng': ['room'], 'lớp': ['class', 'classroom'], 'văn phòng': ['office'],
        
        # Time expressions
        'ngày': ['day'], 'đêm': ['night'], 'sáng': ['morning'], 
        'chiều': ['afternoon'], 'tối': ['evening'],
        'hôm nay': ['today'], 'ngày mai': ['tomorrow'], 'hôm qua': ['yesterday'],
        'cả': ['all', 'whole'], 'suốt': ['all', 'throughout'],
        
        # Objects
        'bài': ['lesson', 'exercise'], 'sách': ['book'], 'vở': ['notebook'],
        'bút': ['pen'], 'xe': ['car', 'vehicle'], 'máy': ['machine'],
        
        # Prepositions
        'ở': ['at', 'in'], 'trong': ['in', 'inside'], 'tại': ['at'],
        'trên': ['on', 'above'], 'dưới': ['under', 'below'],
        'với': ['with'], 'của': ['of'], 'cho': ['for', 'to'],
        
        # Quantities
        'một': ['one', 'a', 'an'], 'hai': ['two'], 'ba': ['three'],
        'nhiều': ['many', 'much'], 'ít': ['few', 'little'],
        'tất cả': ['all'], 'toàn bộ': ['all', 'entire'],
        
        # Adjectives
        'tốt': ['good'], 'xấu': ['bad'], 'đẹp': ['beautiful'], 
        'to': ['big'], 'nhỏ': ['small'], 'nhanh': ['fast'], 'chậm': ['slow'],
        'mới': ['new'], 'cũ': ['old'], 'cao': ['tall', 'high'], 'thấp': ['short', 'low']
    }
    
    score = 0.0
    
    # Direct translation match
    if src_text in core_translations:
        for translation in core_translations[src_text]:
            if translation in trg_text:
                score += 0.4
                break
    
    # Reverse check (English to Vietnamese)
    for vi_word, en_list in core_translations.items():
        if any(en_word in src_text for en_word in en_list) and vi_word in trg_text:
            score += 0.4
            break
    
    # Partial word matching
    src_words = src_text.split()
    trg_words = trg_text.split()
    
    for src_word in src_words:
        if src_word in core_translations:
            for translation in core_translations[src_word]:
                if any(translation in trg_word for trg_word in trg_words):
                    score += 0.2
                    break
    
    # Exact match (for borrowed words, names, numbers)
    if src_text == trg_text:
        score += 0.5
    
    # Length similarity bonus (similar phrase lengths often correlate)
    len_ratio = min(len(src_text), len(trg_text)) / max(len(src_text), len(trg_text))
    if len_ratio > 0.5:
        score += 0.1
    
    return min(score, 0.6)  # Cap at 0.6 for content similarity

def calculate_word_similarity(src_word, trg_word):
    """Calculate similarity between individual words using universal dictionary"""
    src_word = src_word.lower().strip()
    trg_word = trg_word.lower().strip()
    
    # Use the same dynamic translation scoring
    return calculate_dynamic_translation_score(src_word, trg_word)

def align_simple_phrases(src_words, trg_words):
    """Fallback simple phrase alignment"""
    
    # Simple phrase mappings (Vietnamese to English)
    phrase_mappings = {
        # Subject phrases
        ("chúng", "tôi"): ("we",),
        ("tôi",): ("i",),
        ("chúng",): ("we",),
        
        # Verb phrases with aspects
        ("đã", "học"): ("studied",),
        ("đang", "học"): ("studying", "are", "studying"),
        ("sẽ", "học"): ("will", "study"),
        ("học",): ("study", "learn"),
        
        # Object phrases
        ("bài", "học"): ("lesson",),
        ("bài",): ("lesson", "exercise"),
        
        # Location phrases
        ("ở", "trường"): ("at", "school"),
        ("trong", "trường"): ("in", "school"),
        ("tại", "trường"): ("at", "school"),
        ("ở",): ("at", "in"),
        ("trường",): ("school",),
        
        # Time phrases
        ("cả", "ngày"): ("all", "day"),
        ("suốt", "ngày"): ("all", "day"),
        ("ngày",): ("day",),
        ("cả",): ("all", "entire"),
        
        # Common patterns
        ("vào",): ("in", "at"),
        ("với",): ("with",),
        ("và",): ("and",),
        ("của",): ("of", "'s"),
    }
    
    alignment = []
    used_src = set()
    used_trg = set()
    
    # Function to find phrase matches
    def find_phrase_at_position(words, pos, phrases):
        """Find the longest phrase starting at position"""
        best_match = None
        best_length = 0
        
        for phrase in phrases:
            phrase_len = len(phrase)
            if pos + phrase_len <= len(words):
                word_slice = tuple(w.lower() for w in words[pos:pos + phrase_len])
                if word_slice == phrase:
                    if phrase_len > best_length:
                        best_match = phrase
                        best_length = phrase_len
        
        return best_match, best_length
    
    # Phase 1: Multi-word phrase alignment
    src_pos = 0
    while src_pos < len(src_words):
        if src_pos in used_src:
            src_pos += 1
            continue
            
        # Find longest Vietnamese phrase starting at src_pos
        best_vi_phrase, vi_len = find_phrase_at_position(src_words, src_pos, phrase_mappings.keys())
        
        if best_vi_phrase and vi_len > 1:  # Multi-word phrases only
            # Find corresponding English phrase
            en_phrases = phrase_mappings[best_vi_phrase]
            
            # Look for English phrase in target
            found_match = False
            for en_phrase in en_phrases:
                for trg_pos in range(len(trg_words) - len(en_phrase) + 1):
                    if any(trg_pos + i in used_trg for i in range(len(en_phrase))):
                        continue
                        
                    trg_slice = tuple(w.lower() for w in trg_words[trg_pos:trg_pos + len(en_phrase)])
                    if trg_slice == en_phrase:
                        # Create ONE alignment representing the entire phrase
                        # Map the first source word to first target word (as representative)
                        alignment.append([src_pos, trg_pos])
                        
                        # Mark ALL words in the phrase as used
                        for i in range(vi_len):
                            used_src.add(src_pos + i)
                        for i in range(len(en_phrase)):
                            used_trg.add(trg_pos + i)
                        
                        found_match = True
                        break
                if found_match:
                    break
        
        src_pos += 1
    
    # Phase 2: Single word alignment
    for src_pos, src_word in enumerate(src_words):
        if src_pos in used_src:
            continue
            
        src_lower = src_word.lower()
        
        # Check single-word mappings
        for vi_phrase, en_phrases in phrase_mappings.items():
            if len(vi_phrase) == 1 and vi_phrase[0] == src_lower:
                for en_phrase in en_phrases:
                    for trg_pos, trg_word in enumerate(trg_words):
                        if trg_pos not in used_trg and trg_word.lower() in en_phrase:
                            alignment.append([src_pos, trg_pos])
                            used_src.add(src_pos)
                            used_trg.add(trg_pos)
                            break
                    if src_pos in used_src:
                        break
                if src_pos in used_src:
                    break
    
    # Phase 3: Exact matches for remaining words
    for src_pos, src_word in enumerate(src_words):
        if src_pos in used_src:
            continue
            
        for trg_pos, trg_word in enumerate(trg_words):
            if trg_pos not in used_trg and src_word.lower() == trg_word.lower():
                alignment.append([src_pos, trg_pos])
                used_src.add(src_pos)
                used_trg.add(trg_pos)
                break
    
    # Phase 4: Positional alignment for remaining words
    unaligned_src = [i for i in range(len(src_words)) if i not in used_src]
    unaligned_trg = [i for i in range(len(trg_words)) if i not in used_trg]
    
    for i, src_idx in enumerate(unaligned_src):
        if i < len(unaligned_trg):
            alignment.append([src_idx, unaligned_trg[i]])
    
    # Sort alignment by source index
    alignment.sort(key=lambda x: x[0])
    
    return alignment

def swap_alignment_indices(alignments):
    """Swap source and target indices in alignment results"""
    if not alignments:
        return alignments
    
    swapped_alignments = []
    for alignment in alignments:
        if isinstance(alignment, dict):
            # Swap source_indices with target_indices, and source_words with target_words
            swapped_alignment = alignment.copy()
            swapped_alignment['source_indices'] = alignment.get('target_indices', [])
            swapped_alignment['target_indices'] = alignment.get('source_indices', [])
            swapped_alignment['source_words'] = alignment.get('target_words', '')
            swapped_alignment['target_words'] = alignment.get('source_words', '')
            swapped_alignments.append(swapped_alignment)
        else:
            # Handle old format [[src, trg]] -> [[trg, src]]
            if isinstance(alignment, list) and len(alignment) == 2:
                swapped_alignments.append([alignment[1], alignment[0]])
            else:
                swapped_alignments.append(alignment)
    
    return swapped_alignments

def get_word_alignment(src_text, trg_text, src_lang='auto', trg_lang='auto'):
    """Wrapper function for semantic phrase alignment - always display as VI→EN format"""
    
    # Auto-detect which text is Vietnamese and which is English
    vietnamese_chars = set('àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
    
    src_is_vietnamese = any(char in vietnamese_chars for char in src_text.lower())
    trg_is_vietnamese = any(char in vietnamese_chars for char in trg_text.lower())
    
    if src_is_vietnamese and not trg_is_vietnamese:
        # Case 1: Input is VI → EN (keep as is)
        return get_semantic_phrase_alignment_bidirectional(src_text, trg_text, 'vi', 'en')
    elif not src_is_vietnamese and trg_is_vietnamese:
        # Case 2: Input is EN → VI, but display as VI → EN format
        # Always put Vietnamese first in the display
        return get_semantic_phrase_alignment_bidirectional(trg_text, src_text, 'vi', 'en')
    else:
        # Default fallback (assume VI→EN)
        return get_semantic_phrase_alignment_bidirectional(src_text, trg_text, 'vi', 'en')

def get_dependency_tree_html(text, language):
    """Tạo dependency tree HTML"""
    if language == "en":
        try:
            import spacy
            from spacy import displacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            return displacy.render(doc, style="dep", jupyter=False)
        except:
            pass
    
    # Fallback: simple HTML
    return f'''<div style="text-align: center; padding: 20px;">
        <span style="font-size: 16px;">{text}</span><br>
        <small>Dependency tree visualization for {language}</small>
    </div>'''

def main():
    """Hàm chính để xử lý yêu cầu phân tích"""
    import sys
    import warnings
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Text analysis script')
    parser.add_argument('text', help='Text to analyze')
    parser.add_argument('--tab', help='Specific tab to analyze (analysis, translation, view_alignment, dependency_tree)')
    parser.add_argument('--language', help='Language for analysis (vi, en)')
    
    args = parser.parse_args()
    
    text = args.text
    if not text:
        sys.exit(1)
    
    requested_tab = args.tab or 'all'  # Default to full analysis
    requested_language = args.language
    
    # Suppress all warnings
    warnings.filterwarnings("ignore")
    
    # Detect language (simple heuristic)
    vietnamese_chars = set('àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
    is_vietnamese = any(char in vietnamese_chars for char in text.lower())

    # Determine language
    if requested_language:
        is_vietnamese = requested_language == 'vi'
    
    if is_vietnamese:
        # Vietnamese text analysis
        language = "vi"
        original_analysis = get_mock_vietnamese_analysis(text)
        
        # Initialize result with basic analysis (always needed)
        result = {
            "analysis": original_analysis,
            "statistics": get_statistics(original_analysis),
            "language": language,
        }
        
        # Load additional models based on requested tab
        if requested_tab in ['all', 'translation', 'view_alignment']:
            translation = translate_vi_to_en(text)
            result["translation"] = translation
            result["english_translation"] = translation
            result["vietnamese_translation"] = None
            
            if translation and requested_tab in ['all', 'translation']:
                translated_analysis = analyze_english_text(translation)
                result["translated_analysis"] = translated_analysis
                result["english_analysis"] = translated_analysis
                result["vietnamese_analysis"] = []
            
            if translation and requested_tab in ['all', 'view_alignment']:
                result["alignment_result"] = get_word_alignment(text, translation)
        
        if requested_tab in ['all', 'dependency_tree']:
            # For Vietnamese, dependency tree uses English translation
            if 'translation' not in result:
                translation = translate_vi_to_en(text)
                result["translation"] = translation
            result["dependency_tree_html"] = get_dependency_tree_html(result.get("translation"), "en") if result.get("translation") else None
            
    else:
        # English text analysis
        language = "en"
        original_analysis = analyze_english_text(text)
        
        # Initialize result with basic analysis (always needed)
        result = {
            "analysis": original_analysis,
            "statistics": get_statistics(original_analysis),
            "language": language,
        }
        
        # Load additional models based on requested tab
        if requested_tab in ['all', 'translation', 'view_alignment']:
            translation = translate_en_to_vi(text)
            result["translation"] = translation
            result["vietnamese_translation"] = translation
            result["english_translation"] = None
            
            if translation and requested_tab in ['all', 'translation']:
                translated_analysis = get_mock_vietnamese_analysis(translation)
                result["translated_analysis"] = translated_analysis
                result["vietnamese_analysis"] = translated_analysis
                result["english_analysis"] = []
            
            if translation and requested_tab in ['all', 'view_alignment']:
                result["alignment_result"] = get_word_alignment(text, translation)
        
        if requested_tab in ['all', 'dependency_tree']:
            result["dependency_tree_html"] = get_dependency_tree_html(text, "en")
    
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main() 
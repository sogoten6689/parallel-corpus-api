#!/usr/bin/env python3
"""
Enhanced Semantic Alignment using Sentence Transformers
Uses paraphrase-multilingual-mpnet-base-v2 for Vietnamese-English alignment
"""

import sys
import json
import numpy as np
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import torch
except ImportError as e:
    print(f"Missing dependencies: {e}")
    sys.exit(1)

class MultilingualAligner:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        """Initialize with multilingual sentence transformer"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Vietnamese phrase patterns for chunking
        self.vi_patterns = {
            'SUBJECT': [
                ['tôi'], ['chúng', 'tôi'], ['chúng', 'ta'], ['anh'], ['chị'], ['em'],
                ['họ'], ['nó'], ['ông'], ['bà'], ['cô'], ['chú'], ['thầy'],
                ['sinh', 'viên'], ['học', 'sinh'], ['giáo', 'viên'], ['bác', 'sĩ']
            ],
            'VERB': [
                ['đã'], ['đang'], ['sẽ'], ['có', 'thể'], ['cần'], ['muốn'],
                ['học'], ['đi'], ['làm'], ['ăn'], ['uống'], ['ngủ'], ['chơi'],
                ['đọc'], ['viết'], ['nói'], ['nghe'], ['xem'], ['mua'], ['bán']
            ],
            'LOCATION': [
                ['ở'], ['trong'], ['tại'], ['trên'], ['dưới'], ['bên'],
                ['trường'], ['nhà'], ['công', 'ty'], ['văn', 'phòng'],
                ['phòng'], ['lớp'], ['sân'], ['vườn'], ['đường']
            ],
            'TIME': [
                ['hôm', 'nay'], ['ngày', 'mai'], ['hôm', 'qua'],
                ['buổi', 'sáng'], ['buổi', 'chiều'], ['buổi', 'tối'],
                ['cả', 'ngày'], ['suốt', 'ngày'], ['cả', 'đêm'],
                ['lúc'], ['khi'], ['vào'], ['từ'], ['đến']
            ],
            'OBJECT': [
                ['bài'], ['cuốn'], ['quyển'], ['chiếc'], ['cái'], ['con'],
                ['sách'], ['vở'], ['bút'], ['máy'], ['xe'], ['nhà']
            ]
        }

    def extract_phrases(self, text: str, language: str = 'vi') -> List[Dict[str, Any]]:
        """Extract semantic phrases from text"""
        words = text.split()
        phrases = []
        used_positions = set()
        
        if language == 'vi':
            # Vietnamese phrase extraction using patterns
            for phrase_type, pattern_list in self.vi_patterns.items():
                for pattern in pattern_list:
                    pattern_len = len(pattern)
                    for i in range(len(words) - pattern_len + 1):
                        if i in used_positions:
                            continue
                        
                        word_slice = [w.lower() for w in words[i:i + pattern_len]]
                        if word_slice == pattern:
                            phrase_text = ' '.join(words[i:i + pattern_len])
                            phrases.append({
                                'text': phrase_text,
                                'type': phrase_type,
                                'start': i,
                                'end': i + pattern_len - 1,
                                'indices': list(range(i, i + pattern_len))
                            })
                            used_positions.update(range(i, i + pattern_len))
                            break
            
            # Add remaining single words
            for i, word in enumerate(words):
                if i not in used_positions:
                    phrases.append({
                        'text': word,
                        'type': 'WORD',
                        'start': i,
                        'end': i,
                        'indices': [i]
                    })
        
        else:  # English or other languages - simpler word-based approach
            for i, word in enumerate(words):
                phrases.append({
                    'text': word,
                    'type': 'WORD',
                    'start': i,
                    'end': i,
                    'indices': [i]
                })
        
        return sorted(phrases, key=lambda x: x['start'])

    def calculate_semantic_similarity(self, phrase1: str, phrase2: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        try:
            # Get embeddings for both phrases
            embeddings = self.model.encode([phrase1, phrase2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(similarity)
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return 0.0

    def align_phrases_semantic(self, src_text: str, trg_text: str, src_lang: str = 'vi', trg_lang: str = 'en') -> List[Dict[str, Any]]:
        """Align phrases using semantic similarity - supports bidirectional alignment"""
        
        # Extract phrases with dynamic language detection
        src_phrases = self.extract_phrases(src_text, src_lang)
        trg_phrases = self.extract_phrases(trg_text, trg_lang)
        
        if not src_phrases or not trg_phrases:
            return []
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(src_phrases), len(trg_phrases)))
        
        for i, src_phrase in enumerate(src_phrases):
            for j, trg_phrase in enumerate(trg_phrases):
                similarity = self.calculate_semantic_similarity(
                    src_phrase['text'], trg_phrase['text']
                )
                similarity_matrix[i][j] = similarity
        
        # Find best alignments using greedy approach with threshold
        alignments = []
        used_src = set()
        used_trg = set()
        
        # Sort by similarity score (highest first)
        candidates = []
        for i in range(len(src_phrases)):
            for j in range(len(trg_phrases)):
                candidates.append((similarity_matrix[i][j], i, j))
        
        candidates.sort(reverse=True)
        
        # Greedily select best non-overlapping alignments
        similarity_threshold = 0.3  # Adjustable threshold
        
        for similarity, src_idx, trg_idx in candidates:
            if (similarity > similarity_threshold and 
                src_idx not in used_src and 
                trg_idx not in used_trg):
                
                src_phrase = src_phrases[src_idx]
                trg_phrase = trg_phrases[trg_idx]
                
                # Determine semantic type for display
                alignment_type = self.determine_alignment_type(src_phrase, trg_phrase)
                
                alignments.append({
                    'source_words': src_phrase['text'],
                    'target_words': trg_phrase['text'],
                    'source_indices': src_phrase['indices'],
                    'target_indices': trg_phrase['indices'],
                    'type': alignment_type,
                    'similarity_score': float(similarity)
                })
                
                used_src.add(src_idx)
                used_trg.add(trg_idx)
        
        return alignments

    def determine_alignment_type(self, src_phrase: Dict, trg_phrase: Dict) -> str:
        """Determine alignment type for display"""
        src_type = src_phrase['type']
        
        # Type mappings for display
        type_mappings = {
            'SUBJECT': 'Cụm chủ ngữ',
            'VERB': 'Cụm động từ',
            'LOCATION': 'Cụm địa điểm',
            'TIME': 'Cụm thời gian',
            'OBJECT': 'Cụm tân ngữ',
            'WORD': 'Từ đơn'
        }
        
        # Check for multi-word phrases
        if len(src_phrase['indices']) > 1 or len(trg_phrase['indices']) > 1:
            if src_type in type_mappings and src_type != 'WORD':
                return type_mappings[src_type]
            else:
                return 'Cụm từ'
        
        return type_mappings.get(src_type, 'Từ đơn')

def main():
    try:
        # Read input from stdin
        input_data = json.loads(sys.stdin.read())
        src_text = input_data.get('source', '')
        trg_text = input_data.get('target', '')
        src_lang = input_data.get('source_lang', 'vi')
        trg_lang = input_data.get('target_lang', 'en')
        
        if not src_text or not trg_text:
            result = {
                'status': 'error',
                'error': 'Source and target texts are required'
            }
        else:
            # Initialize aligner
            aligner = MultilingualAligner()
            
            # Perform alignment with language specification
            alignments = aligner.align_phrases_semantic(src_text, trg_text, src_lang, trg_lang)
            
            result = {
                'status': 'success',
                'alignments': alignments,
                'total_pairs': len(alignments)
            }
        
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        error_result = {
            'status': 'error',
            'error': str(e)
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

if __name__ == '__main__':
    main() 
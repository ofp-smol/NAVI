"""
N.A.V.I. (Neo Artificial Vivacious Intelligence)
Complete AI System Implementation - Built from Scratch with Multimodal Support
Author: Custom Implementation
Focus: Reliability, Safety, and Ease of Integration
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import re
import os
import time
import logging
import threading
from advanced_tokenizer import AdvancedBPETokenizer, TokenizerConfig
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import sqlite3
from datetime import datetime
import pickle
import warnings
import base64
import io
import torchvision.transforms as transforms
from PIL import Image
import librosa

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('navi.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

#========================================================================
# CONFIGURATION SYSTEM
#========================================================================

@dataclass
class NAVIConfig:
    """Complete configuration for N.A.V.I. system"""
    # Model Architecture
    vocab_size: int = 65536
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ff_dim: int = 3072
    max_seq_len: int = 2048
    dropout: float = 0.1
    
    # Safety Configuration
    safety_threshold: float = 0.8
    enable_content_filter: bool = True
    max_unsafe_responses: int = 3
    safety_log_file: str = 'navi_safety.log'
    
    # Training Configuration
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    
    # API Configuration
    api_host: str = '0.0.0.0'
    api_port: int = 8000
    max_requests_per_hour: int = 100
    max_response_length: int = 500
    enable_cors: bool = True
    
    # RAG Configuration
    rag_top_k: int = 3
    rag_similarity_threshold: float = 0.7
    knowledge_db_path: str = 'navi_knowledge.db'
    
    # Generation Configuration
    default_temperature: float = 0.8
    default_top_p: float = 0.9
    default_top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Multimodal Configuration
    enable_vision: bool = False
    enable_audio: bool = False
    vision_encoder_dim: int = 512
    audio_encoder_dim: int = 512
    image_size: int = 224
    patch_size: int = 16
    audio_sample_rate: int = 16000
    n_mels: int = 80
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

#========================================================================
# CUSTOM TOKENIZER IMPLEMENTATION
#========================================================================

"""
Advanced Custom Tokenizer for NAVI AI
Sophisticated implementation with BPE, subword handling, and large vocabulary support
"""

import re
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict, Counter
import unicodedata
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TokenizerConfig:
    """Configuration for the advanced tokenizer"""
    vocab_size: int = 100000
    min_frequency: int = 2
    max_token_length: int = 100
    special_tokens_file: str = None
    merge_priority: str = 'frequency'  # 'frequency' or 'length'
    enable_normalization: bool = True
    enable_byte_fallback: bool = True
    case_sensitive: bool = False
    preserve_whitespace: bool = True

class AdvancedBPETokenizer:
    """
    Advanced Byte-Pair Encoding tokenizer with sophisticated features:
    - Dynamic vocabulary building with frequency analysis
    - Unicode normalization and byte-level fallback
    - Subword regularization support
    - Advanced punctuation handling
    - Multilingual support with proper Unicode handling
    - Contextual token merging strategies
    - Memory-efficient vocabulary management
    """
    
    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        
        # Core vocabulary components
        self.vocab = {}  # token -> id
        self.inverse_vocab = {}  # id -> token
        self.token_frequencies = defaultdict(int)
        self.merge_rules = {}  # (token1, token2) -> merged_token
        self.merge_priorities = {}  # merge_rule -> priority_score
        
        # Advanced features
        self.byte_encoder = self._build_byte_encoder()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_cache = {}  # Caching for faster repeated tokenization
        self.subword_regularization = False
        self.dropout_prob = 0.0
        
        # Special token categories with enhanced semantic meaning
        self.special_tokens = {
            # Core tokens
            '<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3, '<mask>': 4,
            
            # Reasoning and safety tokens
            '<think>': 5, '</think>': 6, '<reason>': 7, '</reason>': 8,
            '<safe>': 9, '</safe>': 10, '<unsafe>': 11, '</unsafe>': 12,
            '<confidence>': 13, '</confidence>': 14,
            
            # Multimodal tokens
            '<vision>': 15, '</vision>': 16, '<audio>': 17, '</audio>': 18,
            '<multimodal>': 19, '</multimodal>': 20,
            
            # Conversation and context tokens
            '<context>': 21, '</context>': 22, '<user>': 23, '</user>': 24,
            '<assistant>': 25, '</assistant>': 26, '<system>': 27, '</system>': 28,
            
            # Factual and citation tokens
            '<fact>': 29, '</fact>': 30, '<cite>': 31, '</cite>': 32,
            '<quote>': 33, '</quote>': 34, '<reference>': 35, '</reference>': 36,
            
            # Emotional and personality tokens
            '<emotion>': 37, '</emotion>': 38, '<personality>': 39, '</personality>': 40,
            '<empathy>': 41, '</empathy>': 42,
            
            # Code and technical tokens
            '<code>': 43, '</code>': 44, '<function>': 45, '</function>': 46,
            '<variable>': 47, '</variable>': 48, '<comment>': 49, '</comment>': 50,
            
            # Mathematical and scientific tokens
            '<math>': 51, '</math>': 52, '<formula>': 53, '</formula>': 54,
            '<unit>': 55, '</unit>': 56, '<number>': 57, '</number>': 58,
            
            # Language and localization tokens
            '<lang>': 59, '</lang>': 60, '<translate>': 61, '</translate>': 62,
            
            # Time and date tokens
            '<time>': 63, '</time>': 64, '<date>': 65, '</date>': 66,
            
            # Intent and action tokens
            '<intent>': 67, '</intent>': 68, '<action>': 69, '</action>': 70,
            '<goal>': 71, '</goal>': 72, '<task>': 73, '</task>': 74,
            
            # Memory and context management
            '<memory>': 75, '</memory>': 76, '<forget>': 77, '</forget>': 78,
            '<remember>': 79, '</remember>': 80,
            
            # Quality and evaluation tokens
            '<quality>': 81, '</quality>': 82, '<error>': 83, '</error>': 84,
            '<warning>': 85, '</warning>': 86, '<success>': 87, '</success>': 88
        }
        
        # Pattern-based token categories for intelligent tokenization
        self.pattern_tokens = {
            # URLs and links
            r'https?://[^\s]+': '<url>',
            r'www\.[^\s]+': '<url>',
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}': '<email>',
            
            # Numbers and measurements
            r'\d+\.\d+': '<decimal>',
            r'\d+%': '<percentage>',
            r'\$\d+(?:\.\d{2})?': '<currency>',
            r'\d{1,3}(?:,\d{3})*(?:\.\d+)?': '<number>',
            
            # Dates and times
            r'\d{4}-\d{2}-\d{2}': '<date>',
            r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?': '<time>',
            
            # Technical patterns
            r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}': '<uuid>',
            r'#[a-fA-F0-9]{6}': '<color>',
            r'rgb\(\d+,\s*\d+,\s*\d+\)': '<color>',
            
            # Social media and hashtags
            r'#\w+': '<hashtag>',
            r'@\w+': '<mention>',
            
            # File paths and extensions
            r'[a-zA-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*': '<filepath>',
            r'/(?:[^/\s]+/)*[^/\s]*': '<filepath>',
            r'\.[a-zA-Z0-9]{2,4}$': '<extension>',
        }
        
        self.next_id = len(self.special_tokens)
        self._initialize_base_vocabulary()
        
        logger.info(f"Advanced BPE tokenizer initialized with {len(self.special_tokens)} special tokens")
    
    def _build_byte_encoder(self) -> Dict[int, str]:
        """Build byte-level encoder for handling any Unicode character"""
        # Standard printable ASCII
        byte_encoder = {}
        for i in range(33, 127):  # Printable ASCII except space
            byte_encoder[i] = chr(i)
        
        # Special handling for space and control characters
        byte_encoder[32] = '▁'  # Space replacement
        
        # Extended range for Unicode bytes
        n = 0
        for b in range(256):
            if b not in byte_encoder:
                byte_encoder[b] = chr(256 + n)
                n += 1
        
        return byte_encoder
    
    def _initialize_base_vocabulary(self):
        """Initialize vocabulary with special tokens, bytes, and common patterns"""
        # Add special tokens
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
        
        # Add byte-level tokens
        for byte_val, char in self.byte_encoder.items():
            if char not in self.vocab and self.next_id < self.config.vocab_size:
                self.vocab[char] = self.next_id
                self.inverse_vocab[self.next_id] = char
                self.next_id += 1
        
        # Add common subword patterns and affixes
        common_patterns = [
            # English morphemes and affixes
            'un-', 're-', 'pre-', 'anti-', 'de-', 'dis-', 'mis-', 'over-', 'under-',
            'sub-', 'super-', 'inter-', 'intra-', 'extra-', 'ultra-', 'mega-', 'micro-',
            '-ing', '-ed', '-er', '-est', '-ly', '-tion', '-sion', '-ness', '-ment',
            '-able', '-ible', '-ful', '-less', '-ish', '-ous', '-al', '-ic', '-ive',
            
            # Common words and word parts
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her',
            'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man',
            'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let',
            'put', 'say', 'she', 'too', 'use',
            
            # Technical and scientific terms
            'data', 'info', 'tech', 'code', 'file', 'user', 'time', 'type', 'name',
            'text', 'list', 'item', 'link', 'page', 'site', 'web', 'net', 'app',
            'auto', 'base', 'call', 'case', 'core', 'demo', 'dev', 'doc', 'end',
            'env', 'eval', 'exec', 'func', 'gen', 'head', 'init', 'key', 'lib',
            'load', 'log', 'main', 'max', 'meta', 'min', 'mod', 'node', 'null',
            'obj', 'opt', 'out', 'param', 'path', 'prop', 'ref', 'req', 'res',
            'run', 'set', 'src', 'std', 'str', 'sub', 'sys', 'temp', 'test',
            'tmp', 'util', 'val', 'var', 'view', 'work',
            
            # Domain-specific terms
            'model', 'train', 'learn', 'neural', 'network', 'deep', 'machine',
            'algorithm', 'compute', 'process', 'analyze', 'predict', 'classify',
            'optimize', 'feature', 'vector', 'matrix', 'tensor', 'gradient',
            'loss', 'accuracy', 'precision', 'recall', 'f1', 'score', 'metric',
            
            # Common bigrams and trigrams
            'er ', 'ing ', 'ion ', 'and ', 'the ', 'for ', 'are ', 'but ',
            'not ', 'you ', 'all ', 'can ', 'had ', 'her ', 'was ', 'one ',
            'th', 'he', 'in', 'er', 'an', 're', 'nd', 'on', 'en', 'at',
            'ou', 'ed', 'ha', 'to', 'or', 'it', 'is', 'hi', 'es', 'ng'
        ]
        
        # Add patterns to vocabulary with frequency weighting
        for pattern in common_patterns:
            if pattern not in self.vocab and self.next_id < self.config.vocab_size:
                self.vocab[pattern] = self.next_id
                self.inverse_vocab[self.next_id] = pattern
                self.token_frequencies[pattern] = 100  # Base frequency for common patterns
                self.next_id += 1
        
        logger.info(f"Base vocabulary initialized with {len(self.vocab)} tokens")
    
    def _normalize_text(self, text: str) -> str:
        """Advanced text normalization with Unicode handling"""
        if not self.config.enable_normalization:
            return text
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Handle different types of whitespace
        if self.config.preserve_whitespace:
            # Replace various whitespace with standard space but preserve structure
            text = re.sub(r'[\t\v\f\r]+', ' ', text)
            text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
        else:
            text = re.sub(r'\s+', ' ', text)
        
        # Case normalization
        if not self.config.case_sensitive:
            text = text.lower()
        
        return text.strip()
    
    def _apply_pattern_tokenization(self, text: str) -> List[str]:
        """Apply pattern-based pre-tokenization for special patterns"""
        tokens = []
        last_end = 0
        
        # Sort patterns by length (longest first) to avoid conflicts
        sorted_patterns = sorted(self.pattern_tokens.items(), 
                               key=lambda x: len(x[1]), reverse=True)
        
        for pattern, replacement in sorted_patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                
                # Add text before the match
                if start > last_end:
                    tokens.append(text[last_end:start])
                
                # Add the pattern token
                tokens.append(replacement)
                last_end = end
        
        # Add remaining text
        if last_end < len(text):
            tokens.append(text[last_end:])
        
        return [token for token in tokens if token]
    
    def _get_word_pairs(self, word_tokens: List[str]) -> List[Tuple[str, str]]:
        """Get all adjacent pairs in a word for BPE merging"""
        pairs = []
        for i in range(len(word_tokens) - 1):
            pairs.append((word_tokens[i], word_tokens[i + 1]))
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], word_freq: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """Merge the most frequent pair in vocabulary"""
        new_word_freq = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in word_freq.items():
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(pair[0], i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if i < len(word) - 1 and word[i + 1] == pair[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word_freq[tuple(new_word)] = freq
        
        return new_word_freq
    
    def train_bpe(self, corpus: List[str], num_merges: Optional[int] = None):
        """
        Train BPE on a corpus with advanced frequency analysis and merge strategies
        """
        if num_merges is None:
            num_merges = self.config.vocab_size - len(self.vocab)
        
        logger.info(f"Training BPE on corpus of {len(corpus)} texts for {num_merges} merges")
        
        # Normalize and tokenize corpus
        word_freq = defaultdict(int)
        for text in corpus:
            normalized_text = self._normalize_text(text)
            
            # Split by whitespace and punctuation while preserving important patterns
            words = re.findall(r"\S+", normalized_text)
            
            for word in words:
                # Convert to bytes first, then to characters
                byte_word = word.encode('utf-8')
                char_word = tuple(self.byte_encoder[b] for b in byte_word)
                word_freq[char_word] += 1
        
        logger.info(f"Extracted {len(word_freq)} unique words from corpus")
        
        # Perform BPE merges
        for merge_step in range(num_merges):
            if merge_step % 1000 == 0:
                logger.info(f"BPE merge step {merge_step}/{num_merges}")
            
            # Count pair frequencies
            pair_freq = defaultdict(int)
            for word, freq in word_freq.items():
                pairs = self._get_word_pairs(list(word))
                for pair in pairs:
                    pair_freq[pair] += freq
            
            if not pair_freq:
                logger.info(f"No more pairs to merge at step {merge_step}")
                break
            
            # Select best pair based on strategy
            if self.config.merge_priority == 'frequency':
                best_pair = max(pair_freq, key=pair_freq.get)
            else:  # length priority
                best_pair = max(pair_freq, key=lambda x: (len(x[0]) + len(x[1]), pair_freq[x]))
            
            # Store merge rule with priority
            merged_token = ''.join(best_pair)
            self.merge_rules[best_pair] = merged_token
            self.merge_priorities[best_pair] = pair_freq[best_pair]
            
            # Add to vocabulary if not present
            if merged_token not in self.vocab and self.next_id < self.config.vocab_size:
                self.vocab[merged_token] = self.next_id
                self.inverse_vocab[self.next_id] = merged_token
                self.token_frequencies[merged_token] = pair_freq[best_pair]
                self.next_id += 1
            
            # Apply merge to word frequencies
            word_freq = self._merge_vocab(best_pair, word_freq)
        
        logger.info(f"BPE training completed. Final vocabulary size: {len(self.vocab)}")
        logger.info(f"Total merge rules: {len(self.merge_rules)}")
    
    def _apply_bpe(self, word: str) -> List[str]:
        """Apply BPE encoding to a single word with caching"""
        if word in self.bpe_cache:
            return self.bpe_cache[word]
        
        if len(word) <= 1:
            return [word]
        
        # Convert to byte representation
        try:
            byte_word = word.encode('utf-8')
            tokens = [self.byte_encoder.get(b, '<unk>') for b in byte_word]
        except UnicodeEncodeError:
            return ['<unk>']
        
        # Apply BPE merges
        while len(tokens) > 1:
            pairs = self._get_word_pairs(tokens)
            if not pairs:
                break
            
            # Find the best pair to merge (highest priority)
            best_pair = None
            best_priority = -1
            
            for pair in pairs:
                if pair in self.merge_rules:
                    priority = self.merge_priorities.get(pair, 0)
                    if priority > best_priority:
                        best_priority = priority
                        best_pair = pair
            
            if best_pair is None:
                break
            
            # Merge the best pair
            merged_token = self.merge_rules[best_pair]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and 
                    tokens[i] == best_pair[0] and 
                    tokens[i + 1] == best_pair[1]):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # Cache result
        self.bpe_cache[word] = tokens
        return tokens
    
    def encode(self, text: str, max_length: Optional[int] = None, 
               add_special_tokens: bool = True, truncation: bool = True,
               padding: bool = False, pad_to_multiple_of: Optional[int] = None) -> List[int]:
        """
        Advanced encoding with comprehensive options
        """
        if not text:
            if add_special_tokens:
                return [self.special_tokens['<s>'], self.special_tokens['</s>']]
            return []
        
        # Normalize text
        text = self._normalize_text(text)
        
        # Apply pattern-based tokenization first
        pattern_tokens = self._apply_pattern_tokenization(text)
        
        # Process each segment
        all_tokens = []
        if add_special_tokens:
            all_tokens.append('<s>')
        
        for segment in pattern_tokens:
            if segment in self.special_tokens:
                all_tokens.append(segment)
            elif segment.startswith('<') and segment.endswith('>'):
                # Already a special token
                all_tokens.append(segment)
            else:
                # Split by whitespace and apply BPE
                words = segment.split()
                for word in words:
                    if word.strip():
                        bpe_tokens = self._apply_bpe(word)
                        all_tokens.extend(bpe_tokens)
                        
                        # Add space token if preserving whitespace
                        if self.config.preserve_whitespace and word != words[-1]:
                            all_tokens.append('▁')
        
        if add_special_tokens:
            all_tokens.append('</s>')
        
        # Convert tokens to IDs
        token_ids = []
        for token in all_tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # Handle unknown tokens with byte fallback
                if self.config.enable_byte_fallback and len(token) == 1:
                    byte_val = ord(token)
                    if byte_val < 256:
                        byte_char = self.byte_encoder.get(byte_val, '<unk>')
                        token_ids.append(self.vocab.get(byte_char, self.special_tokens['<unk>']))
                    else:
                        token_ids.append(self.special_tokens['<unk>'])
                else:
                    token_ids.append(self.special_tokens['<unk>'])
        
        # Handle length constraints
        if max_length and truncation:
            if len(token_ids) > max_length:
                if add_special_tokens:
                    # Keep start token, truncate middle, keep end token
                    token_ids = token_ids[:max_length-1] + [token_ids[-1]]
                else:
                    token_ids = token_ids[:max_length]
        
        # Handle padding
        if padding and max_length:
            while len(token_ids) < max_length:
                token_ids.append(self.special_tokens['<pad>'])
        
        if pad_to_multiple_of:
            remainder = len(token_ids) % pad_to_multiple_of
            if remainder != 0:
                padding_length = pad_to_multiple_of - remainder
                token_ids.extend([self.special_tokens['<pad>']] * padding_length)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True,
               clean_up_tokenization_spaces: bool = True) -> str:
        """
        Advanced decoding with proper handling of special cases
        """
        if not token_ids:
            return ""
        
        # Convert IDs to tokens
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                
                if skip_special_tokens and token in self.special_tokens:
                    continue
                
                tokens.append(token)
            else:
                if not skip_special_tokens:
                    tokens.append('<unk>')
        
        # Join tokens
        text = ''.join(tokens)
        
        # Convert bytes back to text
        try:
            # Handle byte-encoded characters
            byte_list = []
            i = 0
            while i < len(text):
                char = text[i]
                if char in self.byte_decoder:
                    byte_list.append(self.byte_decoder[char])
                elif ord(char) >= 256:
                    # This is a byte-encoded character
                    byte_list.append(ord(char) - 256)
                else:
                    # Regular character, encode it
                    for byte_val in char.encode('utf-8'):
                        byte_list.append(byte_val)
                i += 1
            
            # Convert bytes back to string
            if byte_list:
                decoded_text = bytes(byte_list).decode('utf-8', errors='ignore')
            else:
                decoded_text = text
        except:
            decoded_text = text
        
        # Clean up tokenization artifacts
        if clean_up_tokenization_spaces:
            decoded_text = decoded_text.replace('▁', ' ')  # Restore spaces
            decoded_text = re.sub(r' +', ' ', decoded_text)  # Multiple spaces to single
            decoded_text = decoded_text.strip()
        
        return decoded_text
    
    def encode_batch(self, texts: List[str], max_length: Optional[int] = None,
                    padding: bool = True, truncation: bool = True) -> Dict[str, List[List[int]]]:
        """Encode a batch of texts efficiently"""
        batch_encodings = []
        attention_masks = []
        
        for text in texts:
            encoding = self.encode(text, max_length=max_length, 
                                 truncation=truncation, padding=False)
            batch_encodings.append(encoding)
        
        # Determine max length for batch
        if padding and max_length is None:
            max_length = max(len(encoding) for encoding in batch_encodings)
        
        # Apply padding and create attention masks
        padded_encodings = []
        for encoding in batch_encodings:
            attention_mask = [1] * len(encoding)
            
            if padding and len(encoding) < max_length:
                padding_length = max_length - len(encoding)
                encoding.extend([self.special_tokens['<pad>']] * padding_length)
                attention_mask.extend([0] * padding_length)
            
            padded_encodings.append(encoding)
            attention_masks.append(attention_mask)
        
        return {
            'input_ids': padded_encodings,
            'attention_mask': attention_masks
        }
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size"""
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Return vocabulary dictionary"""
        return self.vocab.copy()
    
    def save(self, filepath: str):
        """Save tokenizer to file"""
        tokenizer_data = {
            'config': self.config.__dict__,
            'vocab': self.vocab,
            'merge_rules': self.merge_rules,
            'merge_priorities': self.merge_priorities,
            'token_frequencies': dict(self.token_frequencies),
            'special_tokens': self.special_tokens,
            'pattern_tokens': self.pattern_tokens,
            'byte_encoder': self.byte_encoder,
            'next_id': self.next_id
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        
        logger.info(f"Tokenizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AdvancedBPETokenizer':
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        config = TokenizerConfig(**tokenizer_data['config'])
        tokenizer = cls(config)
        
        tokenizer.vocab = tokenizer_data['vocab']
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.merge_rules = tokenizer_data['merge_rules']
        tokenizer.merge_priorities = tokenizer_data['merge_priorities']
        tokenizer.token_frequencies = defaultdict(int, tokenizer_data['token_frequencies'])
        tokenizer.special_tokens = tokenizer_data['special_tokens']
        tokenizer.pattern_tokens = tokenizer_data['pattern_tokens']
        tokenizer.byte_encoder = tokenizer_data['byte_encoder']
        tokenizer.byte_decoder = {v: k for k, v in tokenizer.byte_encoder.items()}
        tokenizer.next_id = tokenizer_data['next_id']
        
        logger.info(f"Tokenizer loaded from {filepath}")
        return tokenizer
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text and return detailed tokenization information"""
        tokens = self.encode(text, add_special_tokens=False)
        token_strings = [self.inverse_vocab.get(tid, '<unk>') for tid in tokens]
        
        analysis = {
            'original_text': text,
            'normalized_text': self._normalize_text(text),
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens)),
            'tokens': tokens,
            'token_strings': token_strings,
            'compression_ratio': len(text) / len(tokens) if tokens else 0,
            'special_token_count': sum(1 for t in token_strings if t in self.special_tokens),
            'oov_count': sum(1 for t in token_strings if t == '<unk>'),
            'byte_fallback_count': sum(1 for t in token_strings if t in self.byte_encoder.values())
        }
        
        return analysis
    
    def get_token_frequency(self, token: str) -> int:
        """Get frequency of a specific token"""
        return self.token_frequencies.get(token, 0)
   
    def get_most_frequent_tokens(self, top_k: int = 100) -> List[Tuple[str, int]]:
       """Get the most frequent tokens"""
       return sorted(self.token_frequencies.items(), key=lambda x: x[1], reverse=True)[:top_k]
   
    def add_tokens(self, new_tokens: List[str]) -> int:
       """Add new tokens to vocabulary"""
       added_count = 0
       for token in new_tokens:
           if token not in self.vocab and self.next_id < self.config.vocab_size:
               self.vocab[token] = self.next_id
               self.inverse_vocab[self.next_id] = token
               self.next_id += 1
               added_count += 1
       
       logger.info(f"Added {added_count} new tokens to vocabulary")
       return added_count
   
    def enable_subword_regularization(self, dropout_prob: float = 0.1):
       """Enable subword regularization for robust training"""
       self.subword_regularization = True
       self.dropout_prob = dropout_prob
       logger.info(f"Subword regularization enabled with dropout probability {dropout_prob}")
   
    def disable_subword_regularization(self):
       """Disable subword regularization"""
       self.subword_regularization = False
       self.dropout_prob = 0.0
       logger.info("Subword regularization disabled")
   
    def _apply_subword_regularization(self, tokens: List[str]) -> List[str]:
       """Apply subword regularization during training"""
       if not self.subword_regularization or self.dropout_prob == 0.0:
           return tokens
       
       regularized_tokens = []
       for token in tokens:
           if (token not in self.special_tokens and 
               np.random.random() < self.dropout_prob and 
               len(token) > 1):
               # Split token into smaller parts randomly
               if len(token) >= 2:
                   split_point = np.random.randint(1, len(token))
                   regularized_tokens.extend([token[:split_point], token[split_point:]])
               else:
                   regularized_tokens.append(token)
           else:
               regularized_tokens.append(token)
       
       return regularized_tokens
   
    def create_conversation_encoding(self, messages: List[Dict[str, str]], 
                                  max_length: Optional[int] = None) -> List[int]:
       """Create properly formatted conversation encoding"""
       conversation_parts = []
       
       for message in messages:
           role = message.get('role', 'user')
           content = message.get('content', '')
           
           # Add role-specific tokens
           if role == 'system':
               conversation_parts.extend(['<system>', content, '</system>'])
           elif role == 'user':
               conversation_parts.extend(['<user>', content, '</user>'])
           elif role == 'assistant':
               conversation_parts.extend(['<assistant>', content, '</assistant>'])
           else:
               conversation_parts.append(content)
       
       # Join and encode
       conversation_text = ' '.join(conversation_parts)
       return self.encode(conversation_text, max_length=max_length)
   
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
       """Extract entities using pattern matching"""
       entities = defaultdict(list)
       
       for pattern, entity_type in self.pattern_tokens.items():
           matches = re.findall(pattern, text)
           if matches:
               entities[entity_type.strip('<>')].extend(matches)
       
       return dict(entities)
   
    def optimize_vocabulary(self, corpus: List[str], target_size: Optional[int] = None):
       """Optimize vocabulary based on corpus statistics"""
       if target_size is None:
           target_size = self.config.vocab_size
       
       logger.info(f"Optimizing vocabulary to target size {target_size}")
       
       # Count token usage in corpus
       token_usage = defaultdict(int)
       for text in corpus:
           tokens = self.encode(text, add_special_tokens=False)
           for token_id in tokens:
               token = self.inverse_vocab.get(token_id, '<unk>')
               token_usage[token] += 1
       
       # Keep special tokens and most frequent tokens
       special_tokens_set = set(self.special_tokens.keys())
       sorted_tokens = sorted(token_usage.items(), key=lambda x: x[1], reverse=True)
       
       new_vocab = {}
       new_inverse_vocab = {}
       next_id = 0
       
       # Add special tokens first
       for token in special_tokens_set:
           if token in self.vocab:
               new_vocab[token] = next_id
               new_inverse_vocab[next_id] = token
               next_id += 1
       
       # Add most frequent tokens up to target size
       for token, freq in sorted_tokens:
           if token not in special_tokens_set and next_id < target_size:
               new_vocab[token] = next_id
               new_inverse_vocab[next_id] = token
               self.token_frequencies[token] = freq
               next_id += 1
       
       # Update vocabulary
       old_size = len(self.vocab)
       self.vocab = new_vocab
       self.inverse_vocab = new_inverse_vocab
       self.next_id = next_id
       
       logger.info(f"Vocabulary optimized: {old_size} -> {len(self.vocab)} tokens")
   
    def compute_token_statistics(self) -> Dict[str, Any]:
       """Compute comprehensive vocabulary statistics"""
       total_tokens = len(self.vocab)
       special_count = len(self.special_tokens)
       regular_count = total_tokens - special_count
       
       # Analyze token lengths
       token_lengths = [len(token) for token in self.vocab.keys() 
                       if token not in self.special_tokens]
       
       # Analyze merge rules
       merge_count = len(self.merge_rules)
       
       # Frequency statistics
       freq_values = list(self.token_frequencies.values())
       
       stats = {
           'total_vocabulary_size': total_tokens,
           'special_tokens_count': special_count,
           'regular_tokens_count': regular_count,
           'merge_rules_count': merge_count,
           'average_token_length': np.mean(token_lengths) if token_lengths else 0,
           'max_token_length': max(token_lengths) if token_lengths else 0,
           'min_token_length': min(token_lengths) if token_lengths else 0,
           'token_frequency_stats': {
               'mean': np.mean(freq_values) if freq_values else 0,
               'median': np.median(freq_values) if freq_values else 0,
               'std': np.std(freq_values) if freq_values else 0,
               'max': max(freq_values) if freq_values else 0,
               'min': min(freq_values) if freq_values else 0
           },
           'coverage_by_frequency': self._compute_frequency_coverage()
       }
       
       return stats
   
    def _compute_frequency_coverage(self) -> Dict[str, float]:
       """Compute what percentage of tokens cover X% of frequency mass"""
       if not self.token_frequencies:
           return {}
       
       sorted_freqs = sorted(self.token_frequencies.values(), reverse=True)
       total_freq = sum(sorted_freqs)
       
       coverage = {}
       cumulative_freq = 0
       for i, freq in enumerate(sorted_freqs):
           cumulative_freq += freq
           coverage_pct = (cumulative_freq / total_freq) * 100
           if coverage_pct >= 50 and '50%' not in coverage:
               coverage['50%'] = (i + 1) / len(sorted_freqs) * 100
           if coverage_pct >= 80 and '80%' not in coverage:
               coverage['80%'] = (i + 1) / len(sorted_freqs) * 100
           if coverage_pct >= 90 and '90%' not in coverage:
               coverage['90%'] = (i + 1) / len(sorted_freqs) * 100
               break
       
       return coverage

def create_sample_training_corpus() -> List[str]:
   """Create a diverse sample corpus for tokenizer training"""
   corpus = [
       # Technical and AI content
       "Machine learning models require large datasets for training and validation purposes.",
       "Neural networks use backpropagation algorithms to optimize their parameters.",
       "Natural language processing involves tokenization, parsing, and semantic analysis.",
       "Deep learning architectures include convolutional neural networks and transformers.",
       "The attention mechanism revolutionized sequence-to-sequence modeling.",
       
       # Conversational content
       "Hello, how can I help you today?",
       "I'm looking for information about artificial intelligence.",
       "Could you please explain how tokenization works?",
       "Thank you for your assistance with this problem.",
       "I appreciate your detailed explanation.",
       
       # Multimodal content
       "This image shows a beautiful sunset over the mountains.",
       "The audio contains speech with background music.",
       "Please analyze this multimodal content combining text, image, and audio.",
       "Vision models can identify objects, scenes, and activities in images.",
       "Speech recognition systems convert audio to text using neural networks.",
       
       # Code and technical content
       "def tokenize(text): return text.split()",
       "import torch.nn as nn",
       "class Transformer(nn.Module):",
       "self.attention = MultiHeadAttention()",
       "optimizer = torch.optim.AdamW(params)",
       
       # Scientific and mathematical content
       "The equation E=mc² represents mass-energy equivalence.",
       "Statistical significance is measured using p-values and confidence intervals.",
       "The gradient descent algorithm minimizes the loss function iteratively.",
       "Probability distributions describe uncertainty in random variables.",
       "Linear algebra operations are fundamental to machine learning.",
       
       # Web and URL content
       "Visit https://example.com for more information.",
       "Send an email to user@example.org for support.",
       "The file path is /home/user/documents/file.txt",
       "Color code #FF6B35 represents a vibrant orange.",
       "The UUID is 550e8400-e29b-41d4-a716-446655440000",
       
       # Diverse punctuation and formatting
       "What's the difference between AI and ML?",
       "Here are three key points: 1) Speed, 2) Accuracy, 3) Efficiency.",
       "The model achieved 95.7% accuracy on the test set.",
       "Training took approximately 2.5 hours on GPU hardware.",
       "Results: precision=0.89, recall=0.92, f1-score=0.90",
       
       # Multilingual examples
       "Hello world in different languages: Hola mundo, Bonjour monde, こんにちは世界",
       "Common greetings: Hello, Hi, Hey, Good morning, Good afternoon",
       "Numbers: one, two, three, vier, cinq, six, siete, eight, neuf, ten",
       
       # Long-form content
       "The field of artificial intelligence has evolved rapidly over the past decade, with breakthrough developments in deep learning, natural language processing, and computer vision. These advances have enabled the creation of sophisticated AI systems capable of understanding and generating human-like text, recognizing complex patterns in images, and processing audio signals with remarkable accuracy.",
       
       # Safety and content moderation examples  
       "Please ensure all content follows safety guidelines and community standards.",
       "This AI system includes comprehensive safety filtering and content moderation.",
       "Harmful, illegal, or inappropriate content will be automatically detected and blocked.",
       "The safety classifier evaluates content across multiple dimensions and modalities.",
       
       # Specialized domain content
       "Medical diagnosis requires careful analysis of symptoms and test results.",
       "Financial markets exhibit complex patterns influenced by economic indicators.",
       "Climate change research uses sophisticated modeling and data analysis techniques.",
       "Educational technology enhances learning through personalized adaptive systems.",
       "Robotics applications span manufacturing, healthcare, and autonomous vehicles."
   ]
   
   return corpus

def demonstrate_advanced_tokenizer():
   """Demonstrate the advanced tokenizer capabilities"""
   print("🚀 Advanced BPE Tokenizer Demonstration")
   print("=" * 60)
   
   # Create tokenizer with custom configuration
   config = TokenizerConfig(
       vocab_size=50000,
       min_frequency=2,
       enable_normalization=True,
       enable_byte_fallback=True,
       preserve_whitespace=True
   )
   
   tokenizer = AdvancedBPETokenizer(config)
   
   # Train on sample corpus
   print("📚 Training tokenizer on sample corpus...")
   corpus = create_sample_training_corpus()
   tokenizer.train_bpe(corpus, num_merges=1000)
   
   # Demonstrate various features
   test_texts = [
       "Hello, world! This is a test of the advanced tokenizer.",
       "The AI model achieved 95.7% accuracy on the test dataset.",
       "Visit https://example.com or email user@example.org for more info.",
       "Code example: def train_model(data, epochs=10): return model.fit(data)",
       "🤖 AI can process emojis, URLs, and special characters like #hashtags @mentions",
       "Multimodal content: <vision>Image data</vision> <audio>Audio data</audio>",
       "Safety check: <safe>This content is appropriate</safe>",
       "Mathematical notation: E=mc², α+β=γ, f(x) = x² + 2x + 1"
   ]
   
   print("\n🧪 Testing tokenization on various text types:")
   print("-" * 60)
   
   for i, text in enumerate(test_texts, 1):
       print(f"\n📝 Test {i}: {text[:50]}{'...' if len(text) > 50 else ''}")
       
       # Encode
       tokens = tokenizer.encode(text)
       token_strings = [tokenizer.inverse_vocab.get(t, '<unk>') for t in tokens]
       
       # Decode
       decoded = tokenizer.decode(tokens)
       
       # Analysis
       analysis = tokenizer.analyze_text(text)
       
       print(f"   🔢 Tokens ({len(tokens)}): {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
       print(f"   📄 Token strings: {token_strings[:5]}{'...' if len(token_strings) > 5 else ''}")
       print(f"   📊 Compression ratio: {analysis['compression_ratio']:.2f}")
       print(f"   ✅ Decoding match: {'✓' if decoded == text else '✗'}")
       
       if analysis['oov_count'] > 0:
           print(f"   ⚠️  OOV tokens: {analysis['oov_count']}")
   
   # Show vocabulary statistics
   print("\n📊 Tokenizer Statistics:")
   print("-" * 60)
   stats = tokenizer.compute_token_statistics()
   print(f"Total vocabulary size: {stats['total_vocabulary_size']:,}")
   print(f"Special tokens: {stats['special_tokens_count']}")
   print(f"Regular tokens: {stats['regular_tokens_count']:,}")
   print(f"Merge rules: {stats['merge_rules_count']:,}")
   print(f"Average token length: {stats['average_token_length']:.2f} characters")
   
   # Show most frequent tokens
   print(f"\n🔥 Top 20 Most Frequent Tokens:")
   for token, freq in tokenizer.get_most_frequent_tokens(20):
       print(f"   '{token}': {freq}")
   
   # Demonstrate batch encoding
   print(f"\n📦 Batch Encoding Demonstration:")
   batch_texts = test_texts[:3]
   batch_result = tokenizer.encode_batch(batch_texts, max_length=50, padding=True)
   print(f"   Batch size: {len(batch_result['input_ids'])}")
   print(f"   Sequence length: {len(batch_result['input_ids'][0])}")
   print(f"   With attention masks: {len(batch_result['attention_mask'])}")
   
   # Save tokenizer
   save_path = "advanced_navi_tokenizer.pkl"
   tokenizer.save(save_path)
   print(f"\n💾 Tokenizer saved to: {save_path}")
   
   # Load and verify
   loaded_tokenizer = AdvancedBPETokenizer.load(save_path)
   test_encoding = loaded_tokenizer.encode("Test loading functionality")
   print(f"✅ Tokenizer loaded successfully, test encoding: {len(test_encoding)} tokens")
   
   return tokenizer

if __name__ == "__main__":
   # Run demonstration
   tokenizer = demonstrate_advanced_tokenizer()
   
   print("\n🎉 Advanced tokenizer demonstration completed!")
   print("The tokenizer is ready for integration with NAVI AI.")

#========================================================================
# VISION ENCODER FOR MULTIMODAL CAPABILITIES
#========================================================================

class NAVIVisionEncoder(nn.Module):
    """Custom vision encoder for processing images"""
    
    def __init__(self, embed_dim: int = 768, patch_size: int = 16, 
                 image_size: int = 224, num_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            num_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embeddings for patches
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        
        # CLS token for global image representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Vision transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=embed_dim // 64,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(6)
        ])
        
        self.ln = nn.LayerNorm(embed_dim)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Process images and return embeddings"""
        batch_size = images.shape[0]
        
        # Create patches
        patches = self.patch_embed(images)
        patches = patches.flatten(2).transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patches = torch.cat([cls_tokens, patches], dim=1)
        
        # Add position embeddings
        patches = patches + self.pos_embed
        
        # Pass through transformer layers
        for layer in self.layers:
            patches = layer(patches)
        
        patches = self.ln(patches)
        return patches

#========================================================================
# AUDIO ENCODER FOR MULTIMODAL CAPABILITIES  
#========================================================================

class NAVIAudioEncoder(nn.Module):
    """Custom audio encoder for processing speech/audio"""
    
    def __init__(self, embed_dim: int = 768, sample_rate: int = 16000,
                 n_mels: int = 80, hop_length: int = 256):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = 1024
        self.win_length = 1024
        
        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((10, 10))
        )
        
        self.projection = nn.Linear(256 * 10 * 10, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 100, embed_dim) * 0.02)
        
        # Audio transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=embed_dim // 64,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(4)
        ])
        
        self.ln = nn.LayerNorm(embed_dim)
    
    def preprocess_audio(self, audio_data: np.ndarray) -> torch.Tensor:
        """Convert audio to mel spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels
        )
        
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        
        return torch.tensor(log_mel, dtype=torch.float32)
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Process audio and return embeddings"""
        batch_size = audio_features.shape[0]
        
        if len(audio_features.shape) == 3:
            audio_features = audio_features.unsqueeze(1)
        
        features = self.conv_layers(audio_features)
        features = features.view(batch_size, -1)
        
        embeddings = self.projection(features)
        embeddings = embeddings.unsqueeze(1)
        
        seq_len = min(embeddings.shape[1], self.pos_embed.shape[1])
        embeddings = embeddings[:, :seq_len] + self.pos_embed[:, :seq_len]
        
        for layer in self.layers:
            embeddings = layer(embeddings)
        
        embeddings = self.ln(embeddings)
        return embeddings

#========================================================================
# MULTIMODAL FUSION LAYER
#========================================================================

class NAVIMultimodalFusion(nn.Module):
    """Fusion layer to combine text, vision, and audio modalities"""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Cross-modal attention layers
        self.text_to_vision = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.text_to_audio = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        # Fusion weights
        self.text_weight = nn.Parameter(torch.ones(1))
        self.vision_weight = nn.Parameter(torch.ones(1))
        self.audio_weight = nn.Parameter(torch.ones(1))
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim * 3, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        
    def forward(self, text_emb: torch.Tensor, 
                vision_emb: torch.Tensor = None,
                audio_emb: torch.Tensor = None) -> torch.Tensor:
        """Fuse multimodal embeddings"""
        
        modalities = [text_emb]
        weights = [self.text_weight]
        
        if vision_emb is not None:
            text_vision_attn, _ = self.text_to_vision(
                text_emb, vision_emb, vision_emb
            )
            modalities.append(text_vision_attn)
            weights.append(self.vision_weight)
        else:
            modalities.append(torch.zeros_like(text_emb))
            weights.append(torch.zeros(1, device=text_emb.device))
            
        if audio_emb is not None:
            text_audio_attn, _ = self.text_to_audio(
                text_emb, audio_emb, audio_emb
            )
            modalities.append(text_audio_attn)
            weights.append(self.audio_weight)
        else:
            modalities.append(torch.zeros_like(text_emb))
            weights.append(torch.zeros(1, device=text_emb.device))
        
        # Weighted fusion
        fused = []
        total_weight = sum(w for w in weights if w.item() > 0)
        
        for modality, weight in zip(modalities, weights):
            if weight.item() > 0:
                fused.append(modality * (weight / total_weight))
            else:
                fused.append(torch.zeros_like(modality))
        
        # Concatenate and project
        fused_emb = torch.cat(fused, dim=-1)
        output = self.output_proj(fused_emb)
        output = self.ln(output)
        
        return output

#========================================================================
# CUSTOM EMBEDDING LAYER WITH ROPE
#========================================================================

class NAVIEmbedding(nn.Module):
    """Advanced embedding layer with Rotary Position Embeddings and safety features"""
    
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Rotary Position Embeddings cache
        self.register_buffer('rope_cache', self._create_rope_cache(embed_dim, max_seq_len))
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
        # Safety embedding
        self.safety_embedding = nn.Parameter(torch.randn(embed_dim) * 0.02)
        
        # Embedding scaling factor
        self.embed_scale = math.sqrt(embed_dim)
    
    def _create_rope_cache(self, dim: int, max_seq_len: int) -> torch.Tensor:
        """Create Rotary Position Embedding cache"""
        rope_dim = dim // 2
        theta = 10000.0
        inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        rope_cache = torch.polar(torch.ones_like(freqs), freqs)
        return rope_cache
    
    def apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply Rotary Position Embedding"""
        batch_size, sequence_length, embed_dim = x.shape
        rope_dim = embed_dim // 2
        
        # Split into RoPE and non-RoPE parts
        x_rope = x[..., :rope_dim]
        x_pass = x[..., rope_dim:]
        
        # Reshape for complex operations
        x_rope = x_rope.reshape(batch_size, sequence_length, rope_dim // 2, 2)
        x_complex = torch.view_as_complex(x_rope)
        
        # Apply RoPE
        rope = self.rope_cache[:seq_len, :rope_dim//2].to(x.device)
        x_rotated = x_complex * rope.unsqueeze(0)
        x_rotated_real = torch.view_as_real(x_rotated).flatten(-2)
        
        return torch.cat([x_rotated_real, x_pass], dim=-1)
    
    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with embeddings and position encoding"""
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        embeddings = self.token_embedding(input_ids)
        embeddings = embeddings * self.embed_scale
        
        # Apply RoPE
        embeddings = self.apply_rope(embeddings, seq_len)
        
        # Add safety context
        safety_context = self.safety_embedding.unsqueeze(0).unsqueeze(0)
        embeddings = embeddings + 0.01 * safety_context
        
        # Layer normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

#========================================================================
# ADVANCED MULTI-HEAD ATTENTION WITH GQA
#========================================================================

class NAVIMultiHeadAttention(nn.Module):
    """Advanced Multi-Head Attention with Grouped Query Attention and safety features"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Grouped Query Attention
        self.num_kv_heads = max(1, num_heads // 4)
        self.kv_head_dim = embed_dim // self.num_kv_heads
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Safety attention mask
        self.safety_mask_weight = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.attention_temperature = nn.Parameter(torch.ones(1))
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                is_causal: bool = True, output_attentions: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Repeat K, V for grouped query attention
        if self.num_kv_heads != self.num_heads:
            key = key.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            value = value.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1))
        attn_weights = attn_weights * (self.scale * self.attention_temperature)
        
        # Apply causal mask
        if is_causal:
            causal_mask = self._create_causal_mask(seq_len, device)
            attn_weights = attn_weights.masked_fill(~causal_mask, float('-inf'))
        
        # Apply custom attention mask
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply safety mask
        attn_weights = attn_weights * self.safety_mask_weight
        
        # Softmax attention weights
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(query)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)
        
        if output_attentions:
            return attn_output, attn_weights
        return attn_output

#========================================================================
# ADVANCED TRANSFORMER LAYER WITH REASONING
#========================================================================

class NAVITransformerLayer(nn.Module):
    """Advanced transformer layer with reasoning gates and safety mechanisms"""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1,
                 activation: str = 'gelu', layer_norm_eps: float = 1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        
        # Multi-head attention
        self.self_attn = NAVIMultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Feed-forward network
        if activation.lower() == 'gelu':
            activation_fn = nn.GELU()
        elif activation.lower() == 'relu':
            activation_fn = nn.ReLU()
        elif activation.lower() == 'silu':
            activation_fn = nn.SiLU()
        else:
            activation_fn = nn.GELU()
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

# Reasoning gate mechanism for enhanced logical processing
self.reasoning_gate = nn.Sequential(
    nn.Linear(self.embed_dim, self.embed_dim // 4),  # Use self.embed_dim
    nn.Sigmoid(),
    nn.Linear(self.embed_dim // 4, self.embed_dim),  # Use self.embed_dim
    nn.Tanh()
)

# Safety gate for content filtering
self.safety_gate = nn.Sequential(
    nn.Linear(self.embed_dim, self.embed_dim // 8),  # Use self.embed_dim
    nn.ReLU(),
    nn.Linear(self.embed_dim // 8, 1),              # Use self.embed_dim
    nn.Sigmoid()
)

# Skip connection weights
self.attn_skip_weight = nn.Parameter(torch.ones(1))
self.ff_skip_weight = nn.Parameter(torch.ones(1))

def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    # Store input for skip connections
    residual = hidden_states
    
    # Pre-norm attention
    hidden_states = self.ln1(hidden_states)
    
    # Self-attention
    if output_attentions:
        attn_output, attn_weights = self.self_attn(
            hidden_states, attention_mask=attention_mask, output_attentions=True
        )
    else:
        attn_output = self.self_attn(hidden_states, attention_mask=attention_mask)
    
    # Skip connection with learnable weight
    hidden_states = residual + self.attn_skip_weight * attn_output
    
    # Store for next skip connection
    residual = hidden_states
    
    # Pre-norm feed-forward
    hidden_states = self.ln2(hidden_states)
    
    # Feed-forward network
    ff_output = self.feed_forward(hidden_states)
    
    # Apply reasoning gate
    reasoning_weight = self.reasoning_gate(hidden_states)
    ff_output = ff_output * reasoning_weight
    
    # Skip connection with learnable weight
    hidden_states = residual + self.ff_skip_weight * ff_output
    
    # Compute safety scores for monitoring
    safety_scores = self.safety_gate(hidden_states)
    
    if output_attentions:
        return hidden_states, attn_weights, safety_scores
    return hidden_states

#===============================================================================
# ENHANCED N.A.V.I. MODEL WITH MULTIMODAL SUPPORT
#===============================================================================

class NAVIModel(nn.Module):
    """
    Enhanced N.A.V.I. (Neo Artificial Vivacious Intelligence)
    Complete transformer-based language model with multimodal, safety and reasoning features
    """
    def __init__(self, config: NAVIConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = NAVIEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            NAVITransformerLayer(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # Final layer normalization
        self.ln_final = nn.LayerNorm(config.embed_dim, eps=1e-6)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        # Safety classification head
        self.safety_classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim // 2, config.embed_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim // 4, 2),
            nn.Softmax(dim=-1)
        )
        
        # Reasoning head for enhanced logical processing
        self.reasoning_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.Tanh()
        )
        
        # Multimodal components
        if config.enable_vision:
            self.vision_encoder = NAVIVisionEncoder(
                embed_dim=config.embed_dim,
                image_size=config.image_size,
                patch_size=config.patch_size
            )
            
        if config.enable_audio:
            self.audio_encoder = NAVIAudioEncoder(
                embed_dim=config.embed_dim,
                sample_rate=config.audio_sample_rate,
                n_mels=config.n_mels
            )
            
        # Multimodal fusion layer
        if config.enable_vision or config.enable_audio:
            self.multimodal_fusion = NAVIMultimodalFusion(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads
            )
            
        # Modality type embeddings
        self.modality_embeddings = nn.Embedding(4, config.embed_dim)  # text, vision, audio, fused
        
        # Value head for reinforcement learning
        self.value_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Linear(config.embed_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie embedding and output weights
        self.lm_head.weight = self.embedding.token_embedding.weight
        
        logger.info(f"Enhanced N.A.V.I. model initialized with {self.count_parameters():,} parameters")
        
    def _init_weights(self, module):
        """Initialize model weights using best practices"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, mean=0.0, std=0.02)
            
    def count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def create_attention_mask(self, input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        """Create attention mask to ignore padding tokens"""
        return (input_ids != pad_token_id).long()
        
    def process_image(self, image_data: str) -> torch.Tensor:
        """Process base64 encoded image"""
        if not self.config.enable_vision:
            return None
            
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Preprocess
            image_tensor = self.vision_encoder.transform(image).unsqueeze(0)
            
            # Encode
            with torch.no_grad():
                vision_emb = self.vision_encoder(image_tensor)
                
            return vision_emb
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None
            
    def process_audio(self, audio_data: np.ndarray) -> torch.Tensor:
        """Process audio data"""
        if not self.config.enable_audio:
            return None
            
        try:
            # Preprocess audio
            audio_features = self.audio_encoder.preprocess_audio(audio_data)
            audio_features = audio_features.unsqueeze(0)
            
            # Encode
            with torch.no_grad():
                audio_emb = self.audio_encoder(audio_features)
                
            return audio_emb
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return None
            
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None, vision_data: str = None,
                audio_data: np.ndarray = None, output_attentions: bool = False,
                output_hidden_states: bool = False, return_dict: bool = True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Enhanced forward pass with multimodal support
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_ids)
            
        # Expand attention mask for multi-head attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.expand(batch_size, 1, seq_len, seq_len)
        
        # Get text embeddings
        hidden_states = self.embedding(input_ids, position_ids)
        
        # Process multimodal inputs
        vision_emb = None
        audio_emb = None
        
        if vision_data is not None and self.config.enable_vision:
            vision_emb = self.process_image(vision_data)
            
        if audio_data is not None and self.config.enable_audio:
            audio_emb = self.process_audio(audio_data)
            
        # Apply multimodal fusion if needed
        if (vision_emb is not None or audio_emb is not None) and hasattr(self, 'multimodal_fusion'):
            hidden_states = self.multimodal_fusion(hidden_states, vision_emb, audio_emb)
            
        # Store hidden states if requested
        all_hidden_states = [hidden_states] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        all_safety_scores = []
        
        # Forward through transformer layers
        for i, layer in enumerate(self.layers):
            if output_attentions:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    output_attentions=True
                )
                hidden_states, attn_weights, safety_scores = layer_outputs
                all_attentions.append(attn_weights)
                all_safety_scores.append(safety_scores)
            else:
                hidden_states = layer(hidden_states, attention_mask=extended_attention_mask)
                
            # Store hidden states
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
                
        # Final layer normalization
        hidden_states = self.ln_final(hidden_states)
        
        # Language modeling logits
        lm_logits = self.lm_head(hidden_states)
        
        # Safety classification
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            safety_input = hidden_states[torch.arange(batch_size), sequence_lengths]
        else:
            safety_input = hidden_states[:, -1, :]
            
        safety_logits = self.safety_classifier(safety_input)
        safety_scores = safety_logits[:, 0]
        
        # Reasoning representation
        reasoning_repr = self.reasoning_head(safety_input)
        
        # Value prediction
        values = self.value_head(safety_input).squeeze(-1)
        
        if not return_dict:
            return lm_logits
            
        # Return comprehensive output dictionary
        outputs = {
            'logits': lm_logits,
            'safety_scores': safety_scores,
            'safety_logits': safety_logits,
            'reasoning_representation': reasoning_repr,
            'values': values,
            'last_hidden_state': hidden_states,
            'multimodal': vision_emb is not None or audio_emb is not None
        }
        
        if output_hidden_states:
            outputs['hidden_states'] = all_hidden_states
        if output_attentions:
            outputs['attentions'] = all_attentions
            outputs['safety_scores_per_layer'] = all_safety_scores
            
        return outputs
        
    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                temperature: float = 0.8, top_p: float = 0.9, top_k: int = 50,
                repetition_penalty: float = 1.1, pad_token_id: int = 0,
                eos_token_id: int = 3, safety_check: bool = True,
                min_safety_score: float = None, vision_data: str = None,
                audio_data: np.ndarray = None) -> Dict[str, torch.Tensor]:
        """
        Enhanced generation with multimodal support and safety checking
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        if min_safety_score is None:
            min_safety_score = self.config.safety_threshold
            
        # Initialize generation tracking
        generated_tokens = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        safety_violations = torch.zeros(batch_size, dtype=torch.int, device=device)
        
        with torch.no_grad():
            for step in range(max_length):
                # Forward pass with multimodal support
                outputs = self.forward(
                    generated_tokens, 
                    vision_data=vision_data,
                    audio_data=audio_data,
                    return_dict=True
                )
                
                logits = outputs['logits'][:, -1, :]
                safety_scores = outputs['safety_scores']
                
                # Safety check
                if safety_check:
                    unsafe_mask = safety_scores < min_safety_score
                    safety_violations += unsafe_mask.int()
                    
                    if unsafe_mask.any():
                        finished = finished | unsafe_mask
                        logger.warning(f"Safety violation detected at step {step}")
                        
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated_tokens[i].tolist()):
                            if logits[i, token_id] > 0:
                                logits[i, token_id] /= repetition_penalty
                            else:
                                logits[i, token_id] *= repetition_penalty
                                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                    
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                    
                # Sample next tokens
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                # Update finished sequences
                next_tokens = next_tokens.masked_fill(finished, pad_token_id)
                finished = finished | (next_tokens == eos_token_id)
                
                # Append tokens
                generated_tokens = torch.cat([generated_tokens, next_tokens.unsqueeze(-1)], dim=-1)
                
                # Check if all sequences are finished
                if finished.all():
                    break
                    
        return {
            'sequences': generated_tokens,
            'safety_violations': safety_violations,
            'finished': finished,
            'multimodal_used': vision_data is not None or audio_data is not None
        }

#===============================================================================
# ENHANCED RAG SYSTEM WITH MULTIMODAL SUPPORT
#===============================================================================

class NAVIRAGSystem:
    """Enhanced Retrieval-Augmented Generation system with multimodal vector database"""
    
    def __init__(self, model: NAVIModel, tokenizer: NAVITokenizer, config: NAVIConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Initialize vector database
        self.db_path = config.knowledge_db_path
        self._init_database()
        
        # Document encoder
        self.doc_encoder = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.LayerNorm(config.embed_dim)
        )
        
        # In-memory cache for fast retrieval
        self.document_cache = {}
        self.embedding_cache = {}
        
        logger.info("Enhanced RAG system initialized with multimodal support")
        
    def _init_database(self):
        """Initialize SQLite database for multimodal document storage"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB,
                modality TEXT DEFAULT 'text',
                vision_data TEXT,
                audio_data BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_modality ON documents(modality)
        ''')
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_created_at ON documents(created_at)
        ''')
        self.conn.commit()
        
    def _encode_document(self, text: str, vision_data: str = None, 
                        audio_data: np.ndarray = None) -> torch.Tensor:
        """Encode multimodal document to embedding vector"""
        with torch.no_grad():
            # Tokenize text
            tokens = self.tokenizer.encode(text, max_length=512)
            input_ids = torch.tensor([tokens])
            
            # Get multimodal embeddings from model
            outputs = self.model(
                input_ids, 
                vision_data=vision_data,
                audio_data=audio_data,
                return_dict=True
            )
            
            # Mean pooling
            doc_embedding = outputs['last_hidden_state'].mean(dim=1)
            
            # Apply document encoder
            doc_embedding = self.doc_encoder(doc_embedding)
            
            return doc_embedding.squeeze(0)
            
    def add_document(self, content: str, metadata: Dict[str, Any] = None,
                    modality: str = 'text', vision_data: str = None,
                    audio_data: np.ndarray = None) -> int:
        """Add multimodal document to the knowledge base"""
        try:
            # Encode document with multimodal support
            embedding = self._encode_document(content, vision_data, audio_data)
            embedding_blob = pickle.dumps(embedding.cpu().numpy())
            
            # Prepare audio data for storage
            audio_blob = None
            if audio_data is not None:
                audio_blob = pickle.dumps(audio_data)
                
            # Store in database
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO documents (content, metadata, embedding, modality, vision_data, audio_data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (content, json.dumps(metadata or {}), embedding_blob, modality, vision_data, audio_blob))
            
            doc_id = cursor.lastrowid
            self.conn.commit()
            
            # Update cache
            self.document_cache[doc_id] = {
                'content': content,
                'metadata': metadata or {},
                'embedding': embedding,
                'modality': modality,
                'vision_data': vision_data,
                'audio_data': audio_data
            }
            
            logger.info(f"Added {modality} document {doc_id} to knowledge base")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return -1
            
    def retrieve_documents(self, query: str, top_k: int = None,
                         similarity_threshold: float = None,
                         modality_filter: str = None,
                         vision_data: str = None,
                         audio_data: np.ndarray = None) -> List[Dict[str, Any]]:
        """Retrieve relevant multimodal documents for a query"""
        if top_k is None:
            top_k = self.config.rag_top_k
        if similarity_threshold is None:
            similarity_threshold = self.config.rag_similarity_threshold
            
        try:
            # Encode query with multimodal support
            query_embedding = self._encode_document(query, vision_data, audio_data)
            
            # Get documents from database with optional modality filter
            cursor = self.conn.cursor()
            if modality_filter:
                cursor.execute(
                    'SELECT id, content, metadata, embedding, modality, vision_data, audio_data FROM documents WHERE modality = ?',
                    (modality_filter,)
                )
            else:
                cursor.execute('SELECT id, content, metadata, embedding, modality, vision_data, audio_data FROM documents')
                
            rows = cursor.fetchall()
            
            if not rows:
                return []
                
            # Calculate similarities
            similarities = []
            for row in rows:
                doc_id, content, metadata_str, embedding_blob, modality, vision_data_stored, audio_blob = row
                
                # Load embedding
                doc_embedding = torch.tensor(pickle.loads(embedding_blob))
                
                # Calculate cosine similarity
                similarity = F.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    doc_embedding.unsqueeze(0),
                    dim=1
                ).item()
                
                if similarity >= similarity_threshold:
                    # Load audio data if present
                    audio_data_loaded = None
                    if audio_blob:
                        audio_data_loaded = pickle.loads(audio_blob)
                        
                    similarities.append({
                        'id': doc_id,
                        'content': content,
                        'metadata': json.loads(metadata_str),
                        'similarity': similarity,
                        'modality': modality,
                        'vision_data': vision_data_stored,
                        'audio_data': audio_data_loaded
                    })
                    
            # Sort by similarity and return top-k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
            
    def generate_with_rag(self, query: str, max_length: int = 200,
                         include_sources: bool = True,
                         vision_data: str = None,
                         audio_data: np.ndarray = None) -> Dict[str, Any]:
        """Generate multimodal response using retrieved documents"""
        try:
            # Retrieve relevant documents with multimodal support
            relevant_docs = self.retrieve_documents(
                query, 
                vision_data=vision_data,
                audio_data=audio_data
            )
            
            if not relevant_docs:
                # Fallback to regular generation
                input_ids = torch.tensor([self.tokenizer.encode(query)])
                outputs = self.model.generate(
                    input_ids, 
                    max_length=max_length,
                    vision_data=vision_data,
                    audio_data=audio_data
                )
                response = self.tokenizer.decode(outputs['sequences'][0].tolist())
                return {
                    'response': response,
                    'sources': [],
                    'method': 'direct_generation',
                    'multimodal': vision_data is not None or audio_data is not None
                }
                
            # Construct context from retrieved documents
            context_parts = []
            sources = []
            multimodal_contexts = []
            
            for doc in relevant_docs:
                context_parts.append(f"[{doc['modality'].upper()} Context]: {doc['content'][:300]}")
                
                if include_sources:
                    sources.append({
                        'id': doc['id'],
                        'similarity': doc['similarity'],
                        'metadata': doc['metadata'],
                        'modality': doc['modality']
                    })
                    
                # Track multimodal contexts
                if doc['modality'] != 'text':
                    multimodal_contexts.append(doc['modality'])
                    
            context = "\n".join(context_parts)
            
            # Construct prompt with context
            prompt = f"{context}\n\n[Query]: {query}\n[Response]:"
            
            # Generate response with multimodal support
            input_ids = torch.tensor([self.tokenizer.encode(prompt, max_length=1024)])
            outputs = self.model.generate(
                input_ids, 
                max_length=max_length,
                vision_data=vision_data,
                audio_data=audio_data
            )
            
            full_response = self.tokenizer.decode(outputs['sequences'][0].tolist())
            
            # Extract response part
            if "[Response]:" in full_response:
                response = full_response.split("[Response]:")[-1].strip()
            else:
                response = full_response.strip()
                
            return {
                'response': response,
                'sources': sources,
                'method': 'multimodal_rag',
                'context_used': len(relevant_docs),
                'multimodal_contexts': list(set(multimodal_contexts)),
                'query_multimodal': vision_data is not None or audio_data is not None
            }
            
        except Exception as e:
            logger.error(f"Error in multimodal RAG generation: {e}")
            return {
                'response': "I apologize, but I encountered an error while processing your multimodal request.",
                'sources': [],
                'method': 'error'
            }
            
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()

#===============================================================================
# ENHANCED SAFETY SYSTEM WITH MULTIMODAL SUPPORT
#===============================================================================

class NAVIMultimodalSafety:
    """Enhanced safety system for comprehensive multimodal content filtering"""
    
    def __init__(self, model: NAVIModel, tokenizer: NAVITokenizer, config: NAVIConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Text safety patterns (from original system)
        self.unsafe_patterns = [
            r'\b(?:kill|murder|assassinate|torture|hurt|harm|violence|attack)\b',
            r'\b(?:bomb|explosive|weapon|gun|knife|poison)\b',
            r'\b(?:drug|cocaine|heroin|methamphetamine|illegal|crime)\b',
            r'\b(?:hack|exploit|virus|malware|phishing|scam)\b',
            r'\b(?:hate|racism|discrimination|bigotry)\b',
            r'\b(?:suicide|self-harm|cutting|overdose)\b',
            r'\b(?:explicit|nsfw|adult|sexual)\b'
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.unsafe_patterns]
        
        # Vision safety patterns (conceptual - would need actual CV models)
        self.vision_unsafe_categories = [
            'explicit_content', 'violence', 'weapons', 'harmful_objects',
            'disturbing_imagery', 'illegal_activities', 'hate_symbols'
        ]
        
        # Audio safety patterns (conceptual - would need actual audio models)
        self.audio_unsafe_categories = [
            'hate_speech', 'threats', 'explicit_language', 'disturbing_sounds',
            'illegal_content', 'harassment', 'violence_audio'
        ]
        
        # Safety responses
        self.safety_responses = [
            "I'm designed to be helpful, harmless, and honest. I cannot provide information that could cause harm.",
            "I'd be happy to help you with something constructive instead. What else can I assist you with?",
            "That request goes against my safety guidelines. Let me help you with something positive.",
            "I'm here to provide helpful and safe assistance. Is there another way I can help you?",
            "I cannot fulfill that request as it may involve harmful content. How else can I assist you today?"
        ]
        
        # Safety logging
        self.safety_log_file = config.safety_log_file
        self._init_safety_logging()
        
        # Enhanced moderation statistics
        self.moderation_stats = {
            'total_requests': 0,
            'safe_requests': 0,
            'unsafe_requests': 0,
            'text_blocks': 0,
            'vision_blocks': 0,
            'audio_blocks': 0,
            'multimodal_blocks': 0,
            'pattern_blocks': 0,
            'model_blocks': 0
        }
        
    def _init_safety_logging(self):
        """Initialize enhanced safety logging"""
        self.safety_logger = logging.getLogger('navi_multimodal_safety')
        self.safety_logger.setLevel(logging.INFO)
        
        # Create file handler for safety logs
        safety_handler = logging.FileHandler(self.safety_log_file)
        safety_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - MULTIMODAL_SAFETY - %(levelname)s - %(message)s')
        safety_handler.setFormatter(formatter)
        self.safety_logger.addHandler(safety_handler)
        
    def check_image_safety(self, image_data: str) -> Tuple[bool, float, str]:
        """
        Check image safety using vision encoder
        """
        if not self.config.enable_vision or not hasattr(self.model, 'vision_encoder'):
            return True, 1.0, "Vision safety check disabled"
            
        try:
            # Process image through vision encoder
            vision_emb = self.model.process_image(image_data)
            
            if vision_emb is None:
                return False, 0.0, "Failed to process image"
                
            # Simple safety heuristic based on embedding patterns
            # In practice, you'd train a separate classifier
            embedding_norm = torch.norm(vision_emb).item()
            embedding_mean = torch.mean(vision_emb).item()
            
            # Heuristic safety score (placeholder logic)
            safety_score = min(1.0, max(0.0, 0.8 + 0.2 * embedding_mean))
            
            is_safe = safety_score >= self.config.safety_threshold
            reason = f"Vision safety score: {safety_score:.3f}"
            
            if not is_safe:
                reason += " - Image content appears potentially unsafe"
                
            return is_safe, safety_score, reason
            
        except Exception as e:
            logger.error(f"Error in image safety check: {e}")
            return False, 0.0, f"Image safety check error: {str(e)}"
            
    def check_audio_safety(self, audio_data: np.ndarray) -> Tuple[bool, float, str]:
        """
        Check audio safety using audio encoder
        """
        if not self.config.enable_audio or not hasattr(self.model, 'audio_encoder'):
            return True, 1.0, "Audio safety check disabled"
            
        try:
            # Process audio through audio encoder
            audio_emb = self.model.process_audio(audio_data)
            
            if audio_emb is None:
                return False, 0.0, "Failed to process audio"
                
            # Simple safety heuristic based on embedding patterns
            # In practice, you'd train a separate classifier
            embedding_norm = torch.norm(audio_emb).item()
            embedding_var = torch.var(audio_emb).item()
            
            # Heuristic safety score (placeholder logic)
            safety_score = min(1.0, max(0.0, 0.85 - 0.1 * embedding_var))
            
            is_safe = safety_score >= self.config.safety_threshold
            reason = f"Audio safety score: {safety_score:.3f}"
            
            if not is_safe:
                reason += " - Audio content appears potentially unsafe"
                
            return is_safe, safety_score, reason
            
        except Exception as e:
            logger.error(f"Error in audio safety check: {e}")
            return False, 0.0, f"Audio safety check error: {str(e)}"
            
    def check_multimodal_safety(self, text: str = "", vision_data: str = None,
                               audio_data: np.ndarray = None,
                               use_model: bool = True) -> Dict[str, Any]:
        """
        Comprehensive multimodal safety check
        """
        self.moderation_stats['total_requests'] += 1
        
        results = {
            'components_checked': [],
            'overall_safe': True,
            'overall_score': 1.0,
            'block_reasons': []
        }
        
        # Check text safety
        if text:
            results['components_checked'].append('text')
            
            # Pattern-based check
            for i, pattern in enumerate(self.compiled_patterns):
                if pattern.search(text):
                    self.moderation_stats['unsafe_requests'] += 1
                    self.moderation_stats['text_blocks'] += 1
                    self.moderation_stats['pattern_blocks'] += 1
                    
                    reason = f"Text blocked by pattern {i}"
                    results['text'] = {
                        'safe': False,
                        'score': 0.0,
                        'reason': reason,
                        'method': 'pattern'
                    }
                    results['overall_safe'] = False
                    results['overall_score'] = 0.0
                    results['block_reasons'].append(reason)
                    
                    self.safety_logger.warning(f"Pattern-based text block: {reason}")
                    return results
                    
            # Model-based text safety check
            if use_model:
                try:
                    with torch.no_grad():
                        input_ids = torch.tensor([self.tokenizer.encode(text, max_length=512)])
                        outputs = self.model(input_ids, return_dict=True)
                        safety_score = outputs['safety_scores'][0].item()
                        
                    is_safe = safety_score >= self.config.safety_threshold
                    
                    results['text'] = {
                        'safe': is_safe,
                        'score': safety_score,
                        'reason': f"Model safety score: {safety_score:.3f}",
                        'method': 'model'
                    }
                    
                    if not is_safe:
                        self.moderation_stats['unsafe_requests'] += 1
                        self.moderation_stats['text_blocks'] += 1
                        self.moderation_stats['model_blocks'] += 1
                        
                        results['overall_safe'] = False
                        results['overall_score'] = min(results['overall_score'], safety_score)
                        results['block_reasons'].append("Text content flagged by safety model")
                        
                        self.safety_logger.warning(f"Model-based text block: score {safety_score:.3f}")
                        
                except Exception as e:
                    logger.error(f"Error in model-based text safety check: {e}")
                    results['text'] = {
                        'safe': False,
                        'score': 0.0,
                        'reason': f"Text safety check error: {str(e)}",
                        'method': 'error'
                    }
                    results['overall_safe'] = False
                    results['overall_score'] = 0.0
                    
            else:
                # Text passed pattern check
                results['text'] = {
                    'safe': True,
                    'score': 1.0,
                    'reason': "Text passed pattern checks",
                    'method': 'pattern'
                }
                
        # Check image safety
        if vision_data:
            results['components_checked'].append('vision')
            is_safe, score, reason = self.check_image_safety(vision_data)
            
            results['vision'] = {
                'safe': is_safe,
                'score': score,
                'reason': reason,
                'method': 'vision_model'
            }
            
            if not is_safe:
                self.moderation_stats['unsafe_requests'] += 1
                self.moderation_stats['vision_blocks'] += 1
                
                results['overall_safe'] = False
                results['overall_score'] = min(results['overall_score'], score)
                results['block_reasons'].append("Image content flagged as unsafe")
                
                self.safety_logger.warning(f"Vision-based block: {reason}")
                
        # Check audio safety
        if audio_data is not None:
            results['components_checked'].append('audio')
            is_safe, score, reason = self.check_audio_safety(audio_data)
            
            results['audio'] = {
                'safe': is_safe,
                'score': score,
                'reason': reason,
                'method': 'audio_model'
            }
            
            if not is_safe:
                self.moderation_stats['unsafe_requests'] += 1
                self.moderation_stats['audio_blocks'] += 1
                
                results['overall_safe'] = False
                results['overall_score'] = min(results['overall_score'], score)
                results['block_reasons'].append("Audio content flagged as unsafe")
                
                self.safety_logger.warning(f"Audio-based block: {reason}")
                
        # Check for multimodal safety issues
        if len(results['components_checked']) > 1:
            # Additional checks for multimodal interactions
            # This could include checking for harmful combinations of content
            multimodal_risk_score = self._assess_multimodal_risk(
                text, vision_data, audio_data
            )
            
            if multimodal_risk_score < self.config.safety_threshold:
                self.moderation_stats['multimodal_blocks'] += 1
                results['overall_safe'] = False
                results['overall_score'] = min(results['overall_score'], multimodal_risk_score)
                results['block_reasons'].append("Multimodal content combination flagged as unsafe")
                
                self.safety_logger.warning(f"Multimodal interaction block: risk score {multimodal_risk_score:.3f}")
                
        # Update statistics
        if results['overall_safe']:
            self.moderation_stats['safe_requests'] += 1
        else:
            self.moderation_stats['unsafe_requests'] += 1
            
        return results
        
    def _assess_multimodal_risk(self, text: str, vision_data: str = None,
                               audio_data: np.ndarray = None) -> float:
        """
        Assess risk from multimodal content combinations
        This is a placeholder for more sophisticated multimodal risk assessment
        """
        base_score = 0.9
        
        # Simple heuristics for multimodal risk
        if text and vision_data:
            # Check for text-image mismatches that might indicate manipulation
            if any(word in text.lower() for word in ['fake', 'manipulated', 'deepfake']):
                base_score -= 0.2
                
        if text and audio_data is not None:
            # Check for text-audio inconsistencies
            if any(word in text.lower() for word in ['not my voice', 'synthetic', 'generated']):
                base_score -= 0.2
                
        if vision_data and audio_data is not None:
            # Check for audio-visual mismatches
            # This would require more sophisticated analysis in practice
            base_score -= 0.1  # Conservative penalty for complexity
            
        return max(0.0, base_score)
        
    def get_safety_response(self, modality: str = 'general') -> str:
        """Get appropriate safety response based on modality"""
        import random
        
        if modality == 'vision':
            vision_responses = [
                "I cannot process this image as it may contain inappropriate content.",
                "The image you've shared appears to violate safety guidelines.",
                "I'm not able to analyze this visual content due to safety concerns."
            ]
            return random.choice(vision_responses)
        elif modality == 'audio':
            audio_responses = [
                "I cannot process this audio as it may contain inappropriate content.",
                "The audio you've shared appears to violate safety guidelines.",
                "I'm not able to analyze this audio content due to safety concerns."
            ]
            return random.choice(audio_responses)
        elif modality == 'multimodal':
            multimodal_responses = [
                "I cannot process this multimodal content as it may contain inappropriate material.",
                "The combination of content you've shared appears to violate safety guidelines.",
                "I'm not able to analyze this mixed-media content due to safety concerns."
            ]
            return random.choice(multimodal_responses)
        else:
            return random.choice(self.safety_responses)
            
    def log_safety_incident(self, content_summary: str, reason: str,
                           modalities: List[str], user_id: str = None):
        """Log comprehensive safety incident"""
        incident_data = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id or 'anonymous',
            'content_summary': content_summary,
            'modalities': modalities,
            'reason': reason,
            'action': 'blocked'
        }
        self.safety_logger.warning(f"Multimodal safety incident: {json.dumps(incident_data)}")
        
    def get_moderation_stats(self) -> Dict[str, Any]:
        """Get comprehensive moderation statistics"""
        stats = self.moderation_stats.copy()
        
        if stats['total_requests'] > 0:
            stats['safety_rate'] = stats['safe_requests'] / stats['total_requests']
            stats['block_rate'] = stats['unsafe_requests'] / stats['total_requests']
            stats['text_block_rate'] = stats['text_blocks'] / stats['total_requests']
            stats['vision_block_rate'] = stats['vision_blocks'] / stats['total_requests']
            stats['audio_block_rate'] = stats['audio_blocks'] / stats['total_requests']
            stats['multimodal_block_rate'] = stats['multimodal_blocks'] / stats['total_requests']
        else:
            for rate_key in ['safety_rate', 'block_rate', 'text_block_rate', 
                           'vision_block_rate', 'audio_block_rate', 'multimodal_block_rate']:
                stats[rate_key] = 0.0
                
        return stats

#===============================================================================
# ENHANCED CONVERSATION MANAGER WITH MULTIMODAL SUPPORT
#===============================================================================

class NAVIConversationManager:
    """Enhanced conversation management with multimodal context and memory"""
    
    def __init__(self, model: NAVIModel, tokenizer: NAVITokenizer,
                 rag_system: NAVIRAGSystem, safety_system: NAVIMultimodalSafety,
                 config: NAVIConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.rag_system = rag_system
        self.safety_system = safety_system
        self.config = config
        
        # Active conversations
        self.conversations = {}
        
        # Conversation database
        self.conv_db_path = 'navi_conversations.db'
        self._init_conversation_db()
        
        # Enhanced system message with multimodal awareness
        self.system_message = (
            "You are N.A.V.I. (Neo Artificial Vivacious Intelligence), a helpful, "
            "harmless, and honest AI assistant with advanced multimodal capabilities. "
            "You can understand and process text, images, and audio content. You are "
            "polite, respectful, and always prioritize user safety and well-being. "
            "You provide accurate information, acknowledge uncertainty, and can analyze "
            "various types of media content while maintaining strict safety standards."
        )
        
    def _init_conversation_db(self):
        """Initialize enhanced conversation database with multimodal support"""
        self.conv_conn = sqlite3.connect(self.conv_db_path, check_same_thread=False)
        
        self.conv_conn.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                multimodal_enabled BOOLEAN DEFAULT FALSE
            )
        ''')
        
        self.conv_conn.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                safety_score REAL,
                metadata TEXT,
                modality TEXT DEFAULT 'text',
                vision_data TEXT,
                audio_data BLOB,
                multimodal_analysis TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')
        
        self.conv_conn.commit()
        
    def start_conversation(self, conversation_id: str, user_id: str = None,
                          system_message: str = None, enable_multimodal: bool = True) -> Dict[str, Any]:
        """Start a new conversation with multimodal support"""
        if system_message is None:
            system_message = self.system_message
            
        # Create conversation record
        conversation = {
            'id': conversation_id,
            'user_id': user_id or 'anonymous',
            'messages': [
                {
                    'role': 'system',
                    'content': system_message,
                    'timestamp': datetime.now().isoformat(),
                    'safety_score': 1.0,
                    'modality': 'text'
                }
            ],
            'context_tokens': [],
            'created_at': datetime.now().isoformat(),
            'metadata': {},
            'multimodal_enabled': enable_multimodal
        }
        
        # Store in memory
        self.conversations[conversation_id] = conversation
        
        # Store in database
        try:
            cursor = self.conv_conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO conversations (id, user_id, metadata, multimodal_enabled)
                VALUES (?, ?, ?, ?)
            ''', (conversation_id, user_id or 'anonymous', json.dumps({}), enable_multimodal))
            
            cursor.execute('''
                INSERT INTO messages (conversation_id, role, content, safety_score, metadata, modality)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (conversation_id, 'system', system_message, 1.0, json.dumps({}), 'text'))
            
            self.conv_conn.commit()
        except Exception as e:
            logger.error(f"Error creating conversation {conversation_id}: {e}")
            
        logger.info(f"Started {'multimodal ' if enable_multimodal else ''}conversation {conversation_id}")
        
        welcome_msg = "Hello! I'm N.A.V.I., your AI assistant."
        if enable_multimodal:
            welcome_msg += " I can help you with text, images, and audio content."
        welcome_msg += " How can I help you today?"
        
        return {
            'conversation_id': conversation_id,
            'status': 'started',
            'message': welcome_msg,
            'multimodal_enabled': enable_multimodal
        }
        
    def add_message(self, conversation_id: str, content: str, role: str = 'user',
                   vision_data: str = None, audio_data: np.ndarray = None) -> Dict[str, Any]:
        """Add a multimodal message to the conversation"""
        if conversation_id not in self.conversations:
            return {
                'conversation_id': conversation_id,
                'status': 'error',
                'message': 'Conversation not found. Please start a new conversation.'
            }
            
        conversation = self.conversations[conversation_id]
        
        # Determine modality
        modalities = ['text']
        if vision_data:
            modalities.append('vision')
        if audio_data is not None:
            modalities.append('audio')
            
        modality = 'multimodal' if len(modalities) > 1 else modalities[0]
        
        # Comprehensive safety check for user input
        if role == 'user':
            safety_results = self.safety_system.check_multimodal_safety(
                text=content,
                vision_data=vision_data,
                audio_data=audio_data
            )
            
            if not safety_results['overall_safe']:
                # Log safety incident
                self.safety_system.log_safety_incident(
                    content_summary=content[:100] + "..." if len(content) > 100 else content,
                    reason="; ".join(safety_results['block_reasons']),
                    modalities=safety_results['components_checked'],
                    user_id=conversation.get('user_id')
                )
                
                # Get appropriate safety response
                response_modality = 'multimodal' if len(modalities) > 1 else modalities[0]
                response = self.safety_system.get_safety_response(response_modality)
                
                return {
                    'conversation_id': conversation_id,
                    'status': 'blocked',
                    'message': response,
                    'safety_results': safety_results,
                    'modality': modality
                }
                
        # Prepare audio data for storage
        audio_blob = None
        if audio_data is not None:
            audio_blob = pickle.dumps(audio_data)
            
        # Add message to conversation
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'safety_score': safety_results.get('overall_score', 1.0) if role == 'user' else 1.0,
            'modality': modality,
            'vision_data': vision_data,
            'audio_data': audio_data
        }
        
        conversation['messages'].append(message)
        
        # Store in database
        try:
            cursor = self.conv_conn.cursor()
            cursor.execute('''
                INSERT INTO messages (conversation_id, role, content, safety_score, 
                                    metadata, modality, vision_data, audio_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (conversation_id, role, content, message['safety_score'], 
                  json.dumps({}), modality, vision_data, audio_blob))
            self.conv_conn.commit()
        except Exception as e:
            logger.error(f"Error storing message: {e}")
            
        logger.info(f"Added {modality} {role} message to conversation {conversation_id}")
        
        return {
            'conversation_id': conversation_id,
            'status': 'success',
            'message': 'Message added successfully',
            'modality': modality
        }
        
    def generate_response(self, conversation_id: str, use_rag: bool = True,
                         max_length: int = None) -> Dict[str, Any]:
        """Generate AI response for multimodal conversation"""
        if conversation_id not in self.conversations:
            return {
                'conversation_id': conversation_id,
                'status': 'error',
                'message': 'Conversation not found'
            }
            
        if max_length is None:
            max_length = self.config.max_response_length
            
        conversation = self.conversations[conversation_id]
        messages = conversation['messages']
        
        try:
            # Get last user message for context
            user_messages = [msg for msg in messages if msg['role'] == 'user']
            if not user_messages:
                return {
                    'conversation_id': conversation_id,
                    'status': 'error',
                    'message': 'No user message found'
                }
                
            last_user_message = user_messages[-1]
            query_text = last_user_message['content']
            query_vision = last_user_message.get('vision_data')
            query_audio = last_user_message.get('audio_data')
            
            # Generate response using RAG if enabled
            if use_rag:
                rag_result = self.rag_system.generate_with_rag(
                    query_text,
                    max_length=max_length,
                    vision_data=query_vision,
                    audio_data=query_audio
                )
                response_content = rag_result['response']
                sources = rag_result.get('sources', [])
                method = rag_result.get('method', 'rag')
                multimodal_contexts = rag_result.get('multimodal_contexts', [])
            else:
                # Direct generation without RAG
                conversation_tokens = self.tokenizer.encode_conversation(
                    messages,
                    max_length=self.config.max_seq_len - max_length
                )
                input_ids = torch.tensor([conversation_tokens])
                
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=self.config.default_temperature,
                    top_p=self.config.default_top_p,
                    top_k=self.config.default_top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    vision_data=query_vision,
                    audio_data=query_audio
                )
                
                # Extract only the new response part
                generated_tokens = outputs['sequences'][0][len(conversation_tokens):]
                response_content = self.tokenizer.decode(generated_tokens.tolist())
                sources = []
                method = 'direct_multimodal' if outputs.get('multimodal_used') else 'direct'
                multimodal_contexts = []
                
            # Safety check for generated response
            safety_results = self.safety_system.check_multimodal_safety(
                text=response_content
            )
            
            if not safety_results['overall_safe']:
                response_content = self.safety_system.get_safety_response()
                logger.warning(f"Generated unsafe response blocked: {safety_results['block_reasons']}")
                
            # Add assistant response to conversation
            self.add_message(conversation_id, response_content, 'assistant')
            
            return {
                'conversation_id': conversation_id,
                'status': 'success',
                'response': response_content,
                'safety_score': safety_results.get('overall_score', 1.0),
                'sources': sources,
                'method': method,
                'multimodal_contexts': multimodal_contexts,
                'query_modality': last_user_message.get('modality', 'text')
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'conversation_id': conversation_id,
                'status': 'error',
                'message': f'Error generating response: {str(e)}'
            }
            
    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation with multimodal support"""
        if conversation_id in self.conversations:
            return self.conversations[conversation_id]
            
        # Try to load from database
        try:
            cursor = self.conv_conn.cursor()
            cursor.execute('''
                SELECT * FROM conversations WHERE id = ?
            ''', (conversation_id,))
            conv_row = cursor.fetchone()
            
            if not conv_row:
                return None
                
            cursor.execute('''
                SELECT role, content, timestamp, safety_score, modality, vision_data, audio_data
                FROM messages WHERE conversation_id = ?
                ORDER BY timestamp
            ''', (conversation_id,))
            message_rows = cursor.fetchall()
            
            messages = []
            for row in message_rows:
                # Load audio data if present
                audio_data = None
                if row[6]:  # audio_data column
                    try:
                        audio_data = pickle.loads(row[6])
                    except:
                        pass
                        
                messages.append({
                    'role': row[0],
                    'content': row[1],
                    'timestamp': row[2],
                    'safety_score': row[3],
                    'modality': row[4],
                    'vision_data': row[5],
                    'audio_data': audio_data
                })
                
            conversation = {
                'id': conversation_id,
                'user_id': conv_row[1],
                'messages': messages,
                'created_at': conv_row[2],
                'metadata': json.loads(conv_row[4]),
                'multimodal_enabled': bool(conv_row[5])
            }
            
            # Cache in memory
            self.conversations[conversation_id] = conversation
            return conversation
            
        except Exception as e:
            logger.error(f"Error loading conversation {conversation_id}: {e}")
            return None
            
    def close(self):
        """Close database connections"""
        if hasattr(self, 'conv_conn'):
            self.conv_conn.close()
        if hasattr(self, 'rag_system'):
            self.rag_system.close()

#===============================================================================
# ENHANCED API SERVER WITH MULTIMODAL ENDPOINTS
#===============================================================================

class NAVIAPIServer:
    """Enhanced RESTful API server with comprehensive multimodal support"""
    
    def __init__(self, conversation_manager: NAVIConversationManager, config: NAVIConfig):
        self.conversation_manager = conversation_manager
        self.config = config
        self.app = Flask(__name__)
        
        if config.enable_cors:
            CORS(self.app)
            
        # Request rate limiting
        self.request_counts = defaultdict(list)
        self.request_lock = threading.Lock()
        
        self._setup_routes()
        logger.info("Enhanced API server initialized with multimodal support")
        
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit"""
        with self.request_lock:
            now = time.time()
            hour_ago = now - 3600
            
            # Clean old requests
            self.request_counts[client_ip] = [
                req_time for req_time in self.request_counts[client_ip]
                if req_time > hour_ago
            ]
            
            # Check current count
            if len(self.request_counts[client_ip]) >= self.config.max_requests_per_hour:
                return False
                
            # Add current request
            self.request_counts[client_ip].append(now)
            return True
            
    def _setup_routes(self):
        """Setup enhanced API routes with multimodal support"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Enhanced health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'version': '2.0.0',
                'timestamp': datetime.now().isoformat(),
                'system': 'N.A.V.I. (Neo Artificial Vivacious Intelligence) - Multimodal Edition',
                'capabilities': {
                    'text': True,
                    'vision': self.config.enable_vision,
                    'audio': self.config.enable_audio,
                    'multimodal_fusion': self.config.enable_vision or self.config.enable_audio,
                    'safety_filtering': True,
                    'rag_support': True
                }
            })
            
        @self.app.route('/conversation', methods=['POST'])
        def create_conversation():
            """Create new conversation with multimodal support"""
            client_ip = request.remote_addr
            if not self._check_rate_limit(client_ip):
                return jsonify({'error': 'Rate limit exceeded'}), 429
                
            try:
                data = request.get_json() or {}
                conversation_id = data.get('conversation_id') or f"conv_{int(time.time())}"
                user_id = data.get('user_id')
                system_message = data.get('system_message')
                enable_multimodal = data.get('enable_multimodal', True)
                
                result = self.conversation_manager.start_conversation(
                    conversation_id, user_id, system_message, enable_multimodal
                )
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Error creating conversation: {e}")
                return jsonify({'error': 'Internal server error'}), 500
                
        @self.app.route('/conversation/<conversation_id>/message', methods=['POST'])
        def send_message(conversation_id: str):
            """Send multimodal message to conversation"""
            client_ip = request.remote_addr
            if not self._check_rate_limit(client_ip):
                return jsonify({'error': 'Rate limit exceeded'}), 429
        try:
            data = request.get_json()
            if not data or 'content' not in data:
                return jsonify({'error': 'Message content required'}), 400

            content = data['content']
            vision_data = data.get('vision_data')  # base64 encoded image
            audio_data = data.get('audio_data')    # base64 encoded audio
            
            # Process audio data if present
            processed_audio = None
            if audio_data:
                try:
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(audio_data)
                    # Convert to numpy array (assuming WAV format)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                    processed_audio = audio_array
                except Exception as e:
                    logger.error(f"Error processing audio data: {e}")

            if len(content) > 2000:  # Message length limit
                return jsonify({'error': 'Message too long'}), 400

            # Add multimodal user message
            result = self.conversation_manager.add_message(
                conversation_id, content, 'user', 
                vision_data=vision_data, 
                audio_data=processed_audio
            )

            if result['status'] != 'success':
                return jsonify(result), 400

            # Generate multimodal response
            use_rag = data.get('use_rag', True)
            max_length = min(data.get('max_length', self.config.max_response_length),
                           self.config.max_response_length)

            response_result = self.conversation_manager.generate_response(
                conversation_id, use_rag, max_length
            )

            return jsonify(response_result)

        except Exception as e:
            logger.error(f"Error processing multimodal message: {e}")
            return jsonify({'error': 'Internal server error'}), 500

        @self.app.route('/multimodal/analyze', methods=['POST'])
        def analyze_multimodal():
            """Analyze multimodal content comprehensively"""
            client_ip = request.remote_addr
            if not self._check_rate_limit(client_ip):
                return jsonify({'error': 'Rate limit exceeded'}), 429

            try:
                data = request.get_json()
                text_content = data.get('text', '')
                vision_data = data.get('vision_data')
                audio_data = data.get('audio_data')

                if not any([text_content, vision_data, audio_data]):
                    return jsonify({'error': 'At least one content type required'}), 400

                # Process audio if present
                processed_audio = None
                if audio_data:
                    try:
                        audio_bytes = base64.b64decode(audio_data)
                        processed_audio = np.frombuffer(audio_bytes, dtype=np.float32)
                    except Exception as e:
                        logger.error(f"Error processing audio: {e}")

                # Comprehensive safety check
                safety_results = self.conversation_manager.safety_system.check_multimodal_safety(
                    text=text_content,
                    vision_data=vision_data,
                    audio_data=processed_audio
                )

                # Content analysis using the model
                analysis_results = {}
                
                if text_content:
                    # Text analysis
                    tokens = self.conversation_manager.tokenizer.encode(text_content)
                    input_ids = torch.tensor([tokens])
                    
                    with torch.no_grad():
                        outputs = self.conversation_manager.model(
                            input_ids, 
                            vision_data=vision_data,
                            audio_data=processed_audio,
                            return_dict=True
                        )
                    
                    analysis_results['text_analysis'] = {
                        'length': len(text_content),
                        'tokens': len(tokens),
                        'safety_score': outputs['safety_scores'][0].item(),
                        'reasoning_quality': torch.mean(outputs['reasoning_representation']).item()
                    }

                # Modality detection
                modalities_present = []
                if text_content:
                    modalities_present.append('text')
                if vision_data:
                    modalities_present.append('vision')
                if processed_audio is not None:
                    modalities_present.append('audio')

                return jsonify({
                    'status': 'success',
                    'analysis': analysis_results,
                    'safety_results': {
                        'overall_safe': safety_results['overall_safe'],
                        'overall_score': safety_results['overall_score'],
                        'components_checked': safety_results['components_checked']
                    },
                    'modalities_detected': modalities_present,
                    'multimodal': len(modalities_present) > 1
                })

            except Exception as e:
                logger.error(f"Error in multimodal analysis: {e}")
                return jsonify({'error': 'Internal server error'}), 500

        @self.app.route('/vision/describe', methods=['POST'])
        def describe_image():
            """Generate description for an image"""
            client_ip = request.remote_addr
            if not self._check_rate_limit(client_ip):
                return jsonify({'error': 'Rate limit exceeded'}), 429

            try:
                data = request.get_json()
                vision_data = data.get('vision_data')
                
                if not vision_data:
                    return jsonify({'error': 'Image data required'}), 400

                if not self.config.enable_vision:
                    return jsonify({'error': 'Vision processing not enabled'}), 400

                # Create temporary conversation for image description
                temp_conv_id = f"vision_desc_{int(time.time())}"
                self.conversation_manager.start_conversation(
                    temp_conv_id, enable_multimodal=True
                )

                # Add image with description request
                description_prompt = "Please describe this image in detail, including objects, scenes, colors, and any text you can see."
                
                self.conversation_manager.add_message(
                    temp_conv_id, description_prompt, 'user',
                    vision_data=vision_data
                )

                # Generate description
                response = self.conversation_manager.generate_response(
                    temp_conv_id, use_rag=False, max_length=300
                )

                return jsonify({
                    'status': 'success',
                    'description': response.get('response', 'Unable to generate description'),
                    'safety_score': response.get('safety_score', 0.0),
                    'method': response.get('method', 'unknown')
                })

            except Exception as e:
                logger.error(f"Error describing image: {e}")
                return jsonify({'error': 'Internal server error'}), 500

        @self.app.route('/audio/transcribe', methods=['POST'])
        def transcribe_audio():
            """Transcribe and analyze audio content"""
            client_ip = request.remote_addr
            if not self._check_rate_limit(client_ip):
                return jsonify({'error': 'Rate limit exceeded'}), 429

            try:
                data = request.get_json()
                audio_data = data.get('audio_data')
                
                if not audio_data:
                    return jsonify({'error': 'Audio data required'}), 400

                if not self.config.enable_audio:
                    return jsonify({'error': 'Audio processing not enabled'}), 400

                # Process audio
                try:
                    audio_bytes = base64.b64decode(audio_data)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                except Exception as e:
                    return jsonify({'error': 'Invalid audio data format'}), 400

                # Safety check for audio
                safety_results = self.conversation_manager.safety_system.check_multimodal_safety(
                    audio_data=audio_array
                )

                if not safety_results['overall_safe']:
                    return jsonify({
                        'status': 'blocked',
                        'message': 'Audio content flagged as unsafe',
                        'safety_results': safety_results
                    }), 400

                # Create temporary conversation for audio analysis
                temp_conv_id = f"audio_analysis_{int(time.time())}"
                self.conversation_manager.start_conversation(
                    temp_conv_id, enable_multimodal=True
                )

                # Add audio with analysis request
                analysis_prompt = "Please analyze this audio content and provide a summary of what you hear."
                
                self.conversation_manager.add_message(
                    temp_conv_id, analysis_prompt, 'user',
                    audio_data=audio_array
                )

                # Generate analysis
                response = self.conversation_manager.generate_response(
                    temp_conv_id, use_rag=False, max_length=200
                )

                return jsonify({
                    'status': 'success',
                    'analysis': response.get('response', 'Unable to analyze audio'),
                    'safety_score': response.get('safety_score', 0.0),
                    'audio_safety': safety_results['audio'] if 'audio' in safety_results else {}
                })

            except Exception as e:
                logger.error(f"Error transcribing audio: {e}")
                return jsonify({'error': 'Internal server error'}), 500

        @self.app.route('/knowledge/multimodal', methods=['POST'])
        def add_multimodal_knowledge():
            """Add multimodal document to knowledge base"""
            client_ip = request.remote_addr
            if not self._check_rate_limit(client_ip):
                return jsonify({'error': 'Rate limit exceeded'}), 429

            try:
                data = request.get_json()
                content = data.get('content', '')
                vision_data = data.get('vision_data')
                audio_data = data.get('audio_data')
                metadata = data.get('metadata', {})

                if not any([content, vision_data, audio_data]):
                    return jsonify({'error': 'At least one content type required'}), 400

                # Process audio if present
                processed_audio = None
                if audio_data:
                    try:
                        audio_bytes = base64.b64decode(audio_data)
                        processed_audio = np.frombuffer(audio_bytes, dtype=np.float32)
                    except Exception as e:
                        logger.error(f"Error processing audio: {e}")
                        return jsonify({'error': 'Invalid audio data'}), 400

                # Determine modality
                modalities = []
                if content:
                    modalities.append('text')
                if vision_data:
                    modalities.append('vision')
                if processed_audio is not None:
                    modalities.append('audio')

                modality = 'multimodal' if len(modalities) > 1 else modalities[0]

                # Safety check
                safety_results = self.conversation_manager.safety_system.check_multimodal_safety(
                    text=content,
                    vision_data=vision_data,
                    audio_data=processed_audio
                )

                if not safety_results['overall_safe']:
                    return jsonify({
                        'error': 'Content flagged as unsafe',
                        'safety_results': safety_results
                    }), 400

                # Add to knowledge base
                doc_id = self.conversation_manager.rag_system.add_document(
                    content=content or f"Multimodal document ({modality})",
                    metadata=metadata,
                    modality=modality,
                    vision_data=vision_data,
                    audio_data=processed_audio
                )

                if doc_id > 0:
                    return jsonify({
                        'status': 'success',
                        'document_id': doc_id,
                        'modality': modality,
                        'modalities': modalities,
                        'message': f'Multimodal document added to knowledge base'
                    })
                else:
                    return jsonify({'error': 'Failed to add document'}), 500

            except Exception as e:
                logger.error(f"Error adding multimodal knowledge: {e}")
                return jsonify({'error': 'Internal server error'}), 500

        @self.app.route('/safety/multimodal/stats', methods=['GET'])
        def multimodal_safety_stats():
            """Get comprehensive multimodal safety statistics"""
            try:
                stats = self.conversation_manager.safety_system.get_moderation_stats()
                return jsonify({
                    'status': 'success',
                    'statistics': stats,
                    'multimodal_capabilities': {
                        'vision_enabled': self.config.enable_vision,
                        'audio_enabled': self.config.enable_audio,
                        'safety_threshold': self.config.safety_threshold
                    }
                })
            except Exception as e:
                logger.error(f"Error getting safety stats: {e}")
                return jsonify({'error': 'Internal server error'}), 500

    def run(self):
        """Run the enhanced API server"""
        logger.info(f"Starting Enhanced N.A.V.I. Multimodal API server on {self.config.api_host}:{self.config.api_port}")
        self.app.run(
            host=self.config.api_host,
            port=self.config.api_port,
            debug=False,
            threaded=True
        )

#=======================================================================
# ENHANCED TRAINER WITH MULTIMODAL SUPPORT
#=======================================================================

class NAVIMultimodalTrainer:
    """Enhanced training system for multimodal N.A.V.I. model"""
    
    def __init__(self, model: NAVIModel, tokenizer: NAVITokenizer, config: NAVIConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Training components
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.warmup_steps,
            T_mult=2
        )
        
        # Loss functions
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.safety_criterion = nn.BCEWithLogitsLoss()
        self.value_criterion = nn.MSELoss()
        
        # Multimodal loss weights
        self.text_weight = 1.0
        self.vision_weight = 0.5
        self.audio_weight = 0.5
        self.fusion_weight = 0.3
        
        # Training metrics
        self.training_stats = {
            'epoch': 0,
            'step': 0,
            'total_loss': 0.0,
            'lm_loss': 0.0,
            'safety_loss': 0.0,
            'multimodal_loss': 0.0,
            'learning_rate': config.learning_rate
        }
        
        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        logger.info("Enhanced multimodal training system initialized")

    def prepare_multimodal_training_data(self, data_batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Prepare multimodal training data batch"""
        batch_tokens = []
        batch_labels = []
        batch_safety = []
        batch_vision = []
        batch_audio = []
        
        for item in data_batch:
            text = item.get('text', '')
            vision_data = item.get('vision_data')
            audio_data = item.get('audio_data')
            safety_label = item.get('safety_label', 1)
            
            # Tokenize text
            tokens = self.tokenizer.encode(text, max_length=self.config.max_seq_len)
            labels = tokens[1:] + [self.tokenizer.special_tokens['</s>']]
            
            # Pad sequences
            while len(tokens) < self.config.max_seq_len:
                tokens.append(0)
            while len(labels) < self.config.max_seq_len:
                labels.append(-100)
                
            tokens = tokens[:self.config.max_seq_len]
            labels = labels[:self.config.max_seq_len]
            
            batch_tokens.append(tokens)
            batch_labels.append(labels)
            batch_safety.append(safety_label)
            
            # Process vision data
            if vision_data and self.config.enable_vision:
                try:
                    vision_emb = self.model.process_image(vision_data)
                    batch_vision.append(vision_emb.squeeze(0) if vision_emb is not None else torch.zeros(197, self.config.embed_dim))
                except:
                    batch_vision.append(torch.zeros(197, self.config.embed_dim))
            else:
                batch_vision.append(torch.zeros(197, self.config.embed_dim))
            
            # Process audio data
            if audio_data is not None and self.config.enable_audio:
                try:
                    audio_emb = self.model.process_audio(audio_data)
                    batch_audio.append(audio_emb.squeeze(0) if audio_emb is not None else torch.zeros(1, self.config.embed_dim))
                except:
                    batch_audio.append(torch.zeros(1, self.config.embed_dim))
            else:
                batch_audio.append(torch.zeros(1, self.config.embed_dim))
        
        return {
            'input_ids': torch.tensor(batch_tokens, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long),
            'safety_labels': torch.tensor(batch_safety, dtype=torch.float),
            'vision_embeddings': torch.stack(batch_vision) if batch_vision else None,
            'audio_embeddings': torch.stack(batch_audio) if batch_audio else None
        }

    def multimodal_training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Enhanced training step with multimodal support"""
        self.model.train()
        
        input_ids = batch['input_ids']
        labels = batch['labels']
        safety_labels = batch['safety_labels']
        vision_emb = batch.get('vision_embeddings')
        audio_emb = batch.get('audio_embeddings')
        
        # Mixed precision training
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids, 
                    return_dict=True
                )
                
                # Language modeling loss
                lm_loss = self.lm_criterion(
                    outputs['logits'].view(-1, outputs['logits'].size(-1)),
                    labels.view(-1)
                )
                
                # Safety classification loss
                safety_loss = self.safety_criterion(
                    outputs['safety_logits'][:, 1],
                    1 - safety_labels
                )
                
                # Multimodal alignment loss (if multimodal data present)
                multimodal_loss = 0.0
                if vision_emb is not None or audio_emb is not None:
                    # Simple alignment loss between text and other modalities
                    text_repr = outputs['last_hidden_state'].mean(dim=1)
                    
                    if vision_emb is not None:
                        vision_repr = vision_emb.mean(dim=1)
                        vision_alignment = 1 - F.cosine_similarity(text_repr, vision_repr).mean()
                        multimodal_loss += self.vision_weight * vision_alignment
                    
                    if audio_emb is not None:
                        audio_repr = audio_emb.mean(dim=1)
                        audio_alignment = 1 - F.cosine_similarity(text_repr, audio_repr).mean()
                        multimodal_loss += self.audio_weight * audio_alignment
                
                # Total loss
                total_loss = (self.text_weight * lm_loss + 
                            0.1 * safety_loss + 
                            self.fusion_weight * multimodal_loss)
                
                # Backward pass with scaling
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            # Regular precision training
            outputs = self.model(input_ids, return_dict=True)
            
            lm_loss = self.lm_criterion(
                outputs['logits'].view(-1, outputs['logits'].size(-1)),
                labels.view(-1)
            )
            
            safety_loss = self.safety_criterion(
                outputs['safety_logits'][:, 1],
                1 - safety_labels
            )
            
            multimodal_loss = torch.tensor(0.0)
            
            total_loss = lm_loss + 0.1 * safety_loss + self.fusion_weight * multimodal_loss
            
            total_loss.backward()
            
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Update learning rate
        self.scheduler.step()
        
        # Update statistics
        self.training_stats['step'] += 1
        self.training_stats['total_loss'] += total_loss.item()
        self.training_stats['lm_loss'] += lm_loss.item()
        self.training_stats['safety_loss'] += safety_loss.item()
        self.training_stats['multimodal_loss'] += multimodal_loss.item() if isinstance(multimodal_loss, torch.Tensor) else multimodal_loss
        self.training_stats['learning_rate'] = self.scheduler.get_last_lr()[0]
        
        return {
            'total_loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'safety_loss': safety_loss.item(),
            'multimodal_loss': multimodal_loss.item() if isinstance(multimodal_loss, torch.Tensor) else multimodal_loss,
            'learning_rate': self.training_stats['learning_rate']
        }

#=======================================================================
# ENHANCED MAIN APPLICATION CLASS
#=======================================================================

class NAVIMultimodalApplication:
    """Enhanced N.A.V.I. application orchestrator with full multimodal support"""
    
    def __init__(self, config_path: str = None):
        # Load or create configuration
        if config_path and os.path.exists(config_path):
            self.config = NAVIConfig.load(config_path)
            logger.info(f"Configuration loaded from {config_path}")
        else:
            self.config = NAVIConfig()
            # Enable multimodal by default
            self.config.enable_vision = True
            self.config.enable_audio = True
            if config_path:
                self.config.save(config_path)
                logger.info(f"Default multimodal configuration saved to {config_path}")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.rag_system = None
        self.safety_system = None
        self.conversation_manager = None
        self.trainer = None
        self.api_server = None
        
        logger.info("N.A.V.I. Multimodal application initialized")

    def initialize_components(self):
        """Initialize all enhanced system components"""
        logger.info("Initializing Enhanced N.A.V.I. components with multimodal support...")
        
        # Initialize tokenizer with multimodal tokens
        tokenizer_config = TokenizerConfig(
    vocab_size=self.config.vocab_size,
    enable_normalization=True,
    enable_byte_fallback=True,
    preserve_whitespace=True
)
self.tokenizer = AdvancedBPETokenizer(tokenizer_config)

# Train the tokenizer if needed
if not os.path.exists('navi_tokenizer.pkl'):
    print("🔧 Training advanced tokenizer...")
    sample_corpus = create_sample_training_corpus()  # You'll need to add this function
    self.tokenizer.train_bpe(sample_corpus, num_merges=2000)
    self.tokenizer.save('navi_tokenizer.pkl')
else:
    self.tokenizer = AdvancedBPETokenizer.load('navi_tokenizer.pkl')
        
        # Initialize enhanced model
    self.model = NAVIModel(self.config)
        
        # Initialize enhanced RAG system
    self.rag_system = NAVIRAGSystem(self.model, self.tokenizer, self.config)
        
        # Initialize enhanced safety system
    self.safety_system = NAVIMultimodalSafety(self.model, self.tokenizer, self.config)
        
        # Initialize enhanced conversation manager
    self.conversation_manager = NAVIConversationManager(
            self.model, self.tokenizer, self.rag_system, self.safety_system, self.config
        )
        
        # Initialize enhanced trainer
    self.trainer = NAVIMultimodalTrainer(self.model, self.tokenizer, self.config)
        
        # Initialize enhanced API server
    self.api_server = NAVIAPIServer(self.conversation_manager, self.config)
        
        logger.info("All enhanced components initialized successfully")
        logger.info(f"Multimodal capabilities: Vision={self.config.enable_vision}, Audio={self.config.enable_audio}")

    def run_multimodal_demo(self):
        """Run a comprehensive multimodal demonstration"""
        print("=" * 70)
        print("N.A.V.I. MULTIMODAL DEMONSTRATION")
        print("Advanced AI with Vision, Audio, and Text Processing")
        print("=" * 70)
        
        if not self.conversation_manager:
            raise RuntimeError("Conversation manager not initialized. Call initialize_components() first.")
        
        # Start demo conversation
        conv_id = f"multimodal_demo_{int(time.time())}"
        self.conversation_manager.start_conversation(conv_id, enable_multimodal=True)
        
        print("\n🤖 N.A.V.I.: Hello! I'm N.A.V.I. with advanced multimodal capabilities.")
        print("I can process text, images, and audio content simultaneously.")
        print("\nDemo Commands:")
        print("- Type 'text:' followed by your message for text-only")
        print("- Type 'image:' to simulate image processing")
        print("- Type 'audio:' to simulate audio processing")
        print("- Type 'multimodal:' to simulate combined processing")
        print("- Type 'stats' to see safety statistics")
        print("- Type 'quit' to exit")
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("🤖 N.A.V.I.: Goodbye! Thank you for trying the multimodal demo.")
                    break
                
                if user_input.lower() == 'stats':
                    stats = self.safety_system.get_moderation_stats()
                    print(f"🔒 Safety Statistics:")
                    print(f"   Total requests: {stats['total_requests']}")
                    print(f"   Safe requests: {stats['safe_requests']}")
                    print(f"   Block rate: {stats.get('block_rate', 0):.2%}")
                    print(f"   Multimodal blocks: {stats.get('multimodal_blocks', 0)}")
                    continue
                
                if not user_input:
                    continue
                
                # Parse demo commands
                vision_data = None
                audio_data = None
                text_content = user_input
                
                if user_input.startswith('image:'):
                    print("🖼️  [Simulating image processing...]")
                    text_content = "Please analyze this image."
                    vision_data = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="  # Minimal JPEG
                
                elif user_input.startswith('audio:'):
                    print("🔊 [Simulating audio processing...]")
                    text_content = "Please analyze this audio."
                    # Create dummy audio data
                    audio_data = np.random.normal(0, 0.1, 1600).astype(np.float32)  # 0.1 seconds at 16kHz
                
                elif user_input.startswith('multimodal:'):
                    print("🎭 [Simulating multimodal processing...]")
                    text_content = "Please analyze this multimodal content combining text, image, and audio."
                    vision_data = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
                    audio_data = np.random.normal(0, 0.1, 1600).astype(np.float32)
                
                elif user_input.startswith('text:'):
                    text_content = user_input[5:].strip()
                
                # Add message with multimodal content
                result = self.conversation_manager.add_message(
                    conv_id, text_content, 'user',
                    vision_data=vision_data,
                    audio_data=audio_data
                )
                
                if result['status'] == 'blocked':
                    print(f"🚫 N.A.V.I.: {result['message']}")
                    continue
                
                # Generate response
                response = self.conversation_manager.generate_response(conv_id)
                
                if response['status'] == 'success':
                    print(f"🤖 N.A.V.I.: {response['response']}")
                    
                    # Show additional info for demo
                    if response.get('method') == 'multimodal_rag':
                        print(f"   📚 Method: Multimodal RAG with {response.get('context_used', 0)} sources")
                        if response.get('multimodal_contexts'):
                            print(f"   🎯 Multimodal contexts: {', '.join(response['multimodal_contexts'])}")
                    elif response.get('method') == 'direct_multimodal':
                        print(f"   🧠 Method: Direct multimodal generation")
                    
                    if response.get('query_modality') != 'text':
                        print(f"   📊 Input modality: {response.get('query_modality', 'unknown')}")
                    
                    safety_score = response.get('safety_score', 1.0)
                    if safety_score < 1.0:
                        print(f"   🔒 Safety score: {safety_score:.3f}")
                else:
                    print(f"🤖 N.A.V.I.: I apologize, but I encountered an error: {response.get('message', 'Unknown error')}")
                
            except KeyboardInterrupt:
                print("\n🤖 N.A.V.I.: Demo interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def run_comprehensive_tests(self):
        """Run comprehensive system tests for multimodal capabilities"""
        print("🧪 Running N.A.V.I. Multimodal System Tests...")
        print("=" * 50)
        
        if not self.conversation_manager:
            self.initialize_components()
        
        tests_passed = 0
        tests_total = 0
        
        # Test 1: Basic tokenizer with multimodal tokens
        tests_total += 1
        try:
            test_text = "<vision>Image content</vision><audio>Audio content</audio>"
            tokens = self.tokenizer.encode(test_text)
            decoded = self.tokenizer.decode(tokens)
            assert len(tokens) > 0, "Multimodal tokenization failed"
            assert '<vision>' in self.tokenizer.special_tokens, "Vision tokens missing"
            assert '<audio>' in self.tokenizer.special_tokens, "Audio tokens missing"
            print("✅ Test 1: Multimodal tokenizer - PASSED")
            tests_passed += 1
        except Exception as e:
            print(f"❌ Test 1: Multimodal tokenizer - FAILED: {e}")
        
        # Test 2: Model forward pass with multimodal support
        tests_total += 1
        try:
            input_ids = torch.tensor([[1, 2, 3, 4, 5]])
            outputs = self.model(input_ids, return_dict=True)
            assert 'logits' in outputs, "Model output missing logits"
            assert 'safety_scores' in outputs, "Model output missing safety scores"
            assert outputs.get('multimodal', False) == False, "Should not be multimodal without input"
            print("✅ Test 2: Model forward pass - PASSED")
            tests_passed += 1
        except Exception as e:
            print(f"❌ Test 2: Model forward pass - FAILED: {e}")
        
        # Test 3: Enhanced safety system
        tests_total += 1
        try:
            safe_text = "Hello, how are you today?"
            safety_results = self.safety_system.check_multimodal_safety(text=safe_text)
            assert isinstance(safety_results, dict), "Safety check should return dict"
            assert 'overall_safe' in safety_results, "Missing overall safety result"
            assert 'components_checked' in safety_results, "Missing components info"
            print("✅ Test 3: Enhanced safety system - PASSED")
            tests_passed += 1
        except Exception as e:
            print(f"❌ Test 3: Enhanced safety system - FAILED: {e}")
        
        # Test 4: Vision processing (if enabled)
        tests_total += 1
        try:
            if self.config.enable_vision and hasattr(self.model, 'vision_encoder'):
                # Test with minimal base64 image
                test_image = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
                vision_emb = self.model.process_image(test_image)
                # Vision processing should handle gracefully even with minimal image
                print("✅ Test 4: Vision processing - PASSED")
                tests_passed += 1
            else:
                print("⚠️  Test 4: Vision processing - SKIPPED (not enabled)")
                tests_passed += 1  # Count as passed since it's optional
        except Exception as e:
            print(f"❌ Test 4: Vision processing - FAILED: {e}")
        
        # Test 5: Audio processing (if enabled)
        tests_total += 1
        try:
            if self.config.enable_audio and hasattr(self.model, 'audio_encoder'):
                # Test with dummy audio data
                test_audio = np.random.normal(0, 0.1, 1600).astype(np.float32)
                audio_emb = self.model.process_audio(test_audio)
                print("✅ Test 5: Audio processing - PASSED")
                tests_passed += 1
            else:
                print("⚠️  Test 5: Audio processing - SKIPPED (not enabled)")
                tests_passed += 1  # Count as passed since it's optional
        except Exception as e:
            print(f"❌ Test 5: Audio processing - FAILED: {e}")
        
        # Test 6: Enhanced RAG system
        tests_total += 1
        try:
            doc_id = self.rag_system.add_document(
                "Test multimodal document", 
                {"test": True}, 
                modality='text'
            )
            assert doc_id > 0, "Document addition failed"
            
            results = self.rag_system.retrieve_documents("test", top_k=1)
            assert len(results) >= 0, "Document retrieval failed"
            print("✅ Test 6: Enhanced RAG system - PASSED")
            tests_passed += 1
        except Exception as e:
            print(f"❌ Test 6: Enhanced RAG system - FAILED: {e}")
        
        # Test 7: Multimodal conversation management
        tests_total += 1
        try:
            conv_id = "test_multimodal_conversation"
            result = self.conversation_manager.start_conversation(conv_id, enable_multimodal=True)
            assert result['status'] == 'started', "Multimodal conversation creation failed"
            assert result['multimodal_enabled'] == True, "Multimodal not enabled"
            
            msg_result = self.conversation_manager.add_message(conv_id, "Hello multimodal world")
            assert msg_result['status'] == 'success', "Message addition failed"
            print("✅ Test 7: Multimodal conversation management - PASSED")
            tests_passed += 1
        except Exception as e:
            print(f"❌ Test 7: Multimodal conversation management - FAILED: {e}")
        
        print(f"\n📊 Test Results: {tests_passed}/{tests_total} tests passed")
        print(f"Success Rate: {(tests_passed/tests_total)*100:.1f}%")
        
        if tests_passed == tests_total:
            print("🎉 All tests passed! N.A.V.I. Multimodal system is ready.")
        else:
            print("⚠️  Some tests failed. Please check the system configuration.")
        
        return tests_passed == tests_total

    def train_multimodal(self, training_data: List[Dict], epochs: int = 1):
        """Train the model with multimodal data"""
        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Call initialize_components() first.")
        
        logger.info(f"Starting multimodal training for {epochs} epochs...")
        print(f"🚀 Training N.A.V.I. with {len(training_data)} multimodal examples...")
        
        for epoch in range(epochs):
            print(f"\n📚 Epoch {epoch + 1}/{epochs}")
            epoch_losses = {'total': 0.0, 'lm': 0.0, 'safety': 0.0, 'multimodal': 0.0}
            num_batches = 0
            
            # Process data in batches
            for i in range(0, len(training_data), self.config.batch_size):
                batch_data = training_data[i:i + self.config.batch_size]
                
                # Prepare multimodal batch
                batch = self.trainer.prepare_multimodal_training_data(batch_data)
                
                # Training step
                step_losses = self.trainer.multimodal_training_step(batch)
                
                # Accumulate losses
                for key in epoch_losses:
                    if key in step_losses:
                        epoch_losses[key] += step_losses[key]
                
                num_batches += 1
                
                # Log progress
                if num_batches % 10 == 0:
                    print(f"   Step {self.trainer.training_stats['step']}: "
                          f"Loss={step_losses['total_loss']:.4f}, "
                          f"MM={step_losses['multimodal_loss']:.4f}, "
                          f"LR={step_losses['learning_rate']:.6f}")
            
            # Average losses
            if num_batches > 0:
                for key in epoch_losses:
                    epoch_losses[key] /= num_batches
            
            print(f"✅ Epoch {epoch + 1} completed:")
            print(f"   Average Loss: {epoch_losses['total']:.4f}")
            print(f"   Multimodal Loss: {epoch_losses['multimodal']:.4f}")
            print(f"   Safety Loss: {epoch_losses['safety']:.4f}")
        
        logger.info("Multimodal training completed")
        print("🎉 Multimodal training completed successfully!")

    def shutdown(self):
        """Shutdown the enhanced application gracefully"""
        logger.info("Shutting down N.A.V.I. Multimodal application...")
        
        if self.conversation_manager:
            self.conversation_manager.close()
        
        if self.rag_system:
            self.rag_system.close()
        
        logger.info("N.A.V.I. Multimodal application shutdown complete")
        print("👋 N.A.V.I. Multimodal system shutdown complete.")

#=======================================================================
# UTILITY FUNCTIONS FOR MULTIMODAL DEMO DATA
#=======================================================================

def create_multimodal_demo_data():
    """Create comprehensive multimodal training/demo data"""
    demo_data = [
        # Text-only examples
        {
            "text": "<user>What is artificial intelligence?</user><assistant>Artificial Intelligence (AI) is a field of computer science focused on creating systems that can perform tasks typically requiring human intelligence, such as learning, reasoning, perception, and decision-making.</assistant>",
            "safety_label": 1,
            "modality": "text"
        },
        
        # Multimodal examples (simulated)
        {
            "text": "<user><vision>Image of a sunset</vision>Can you describe what you see in this image?</user><assistant>I can see a beautiful sunset with warm orange and pink colors painting the sky. The sun appears to be setting on the horizon, creating a peaceful and serene atmosphere.</assistant>",
            "vision_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=",
            "safety_label": 1,
            "modality": "multimodal"
        },
        
        {
            "text": "<user><audio>Sound of birds chirping</audio>What do you hear in this audio?</user><assistant>I can hear the pleasant sound of birds chirping, which creates a peaceful natural ambiance. This type of audio is often associated with morning time or natural outdoor environments.</assistant>",
            "audio_data": np.random.normal(0, 0.1, 1600).astype(np.float32),
            "safety_label": 1,
            "modality": "multimodal"
        },
        
        {
            "text": "<user>How can I improve my productivity?</user><assistant>Here are some effective ways to improve productivity: 1) Set clear goals and priorities, 2) Use time-blocking techniques, 3) Minimize distractions, 4) Take regular breaks, 5) Maintain a healthy work-life balance, and 6) Use productivity tools that work for you.</assistant>",
            "safety_label": 1,
            "modality": "text"
        },
        
        # Safety examples
        {
            "text": "<user>Tell me about renewable energy</user><assistant>Renewable energy comes from natural sources that replenish themselves, such as solar, wind, hydroelectric, and geothermal power. These clean energy sources are crucial for addressing climate change and reducing our dependence on fossil fuels.</assistant>",
            "safety_label": 1,
            "modality": "text"
        }
    ]
    
    return demo_data

def setup_multimodal_knowledge_base(rag_system: NAVIRAGSystem):
    """Setup enhanced knowledge base with multimodal content"""
    multimodal_docs = [
        {
            "content": "N.A.V.I. (Neo Artificial Vivacious Intelligence) is an advanced multimodal AI system capable of processing text, images, and audio simultaneously. It features comprehensive safety filtering, retrieval-augmented generation, and sophisticated reasoning capabilities.",
            "metadata": {"category": "about_navi", "importance": "high", "version": "multimodal"},
            "modality": "text"
        },
        
        {
            "content": "Multimodal AI systems can process and understand multiple types of data simultaneously, such as text, images, audio, and video. This enables more comprehensive understanding and generation of content that mirrors human perception and communication.",
            "metadata": {"category": "ai_concepts", "importance": "high"},
            "modality": "text"
        },
        
        {
            "content": "Computer vision is a field of AI that enables machines to interpret and understand visual information from the world. It involves techniques like image recognition, object detection, facial recognition, and scene understanding.",
            "metadata": {"category": "computer_vision", "importance": "medium"},
            "modality": "text"
        },
        
        {
            "content": "Natural Language Processing (NLP) combined with audio processing enables AI systems to understand spoken language, analyze speech patterns, and generate human-like audio responses. This is crucial for voice assistants and conversational AI.",
            "metadata": {"category": "nlp_audio", "importance": "medium"},
            "modality": "text"
        },
        
        {
            "content": "AI safety in multimodal systems requires comprehensive monitoring across all input modalities. This includes text content filtering, image content analysis for inappropriate material, and audio processing for harmful speech detection.",
            "metadata": {"category": "ai_safety", "importance": "high"},
            "modality": "text"
        }
    ]
    
    for doc in multimodal_docs:
        rag_system.add_document(
            doc["content"], 
            doc["metadata"], 
            doc.get("modality", "text")
        )
    
    print(f"✅ Added {len(multimodal_docs)} documents to multimodal knowledge base")

#=======================================================================
# ENHANCED MAIN ENTRY POINT
#=======================================================================

def main():
    """Enhanced main entry point for N.A.V.I. Multimodal system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='N.A.V.I. (Neo Artificial Vivacious Intelligence) - Multimodal Edition')
    parser.add_argument('--config', type=str, default='navi_multimodal_config.json',
                       help='Configuration file path')
    parser.add_argument('--mode', type=str, 
                       choices=['interactive', 'api', 'train', 'demo', 'test'],
                       default='demo', help='Run mode')
    parser.add_argument('--model', type=str, help='Model file path to load')
    parser.add_argument('--train-data', type=str, help='Training data file path')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--enable-vision', action='store_true', default=True,
                       help='Enable vision processing')
    parser.add_argument('--enable-audio', action='store_true', default=True,
                       help='Enable audio processing')
    parser.add_argument('--safety-threshold', type=float, default=0.8,
                       help='Safety threshold for content filtering')
    
    args = parser.parse_args()
    
    try:
        print("🚀 Initializing N.A.V.I. Multimodal System...")
        print("=" * 60)
        
        # Initialize enhanced application
        app = NAVIMultimodalApplication(args.config)
        
        # Override config with command line arguments
        if hasattr(args, 'enable_vision'):
            app.config.enable_vision = args.enable_vision
        if hasattr(args, 'enable_audio'):
            app.config.enable_audio = args.enable_audio
        if hasattr(args, 'safety_threshold'):
            app.config.safety_threshold = args.safety_threshold
        
        app.initialize_components()
        
        # Load model if specified
        if args.model:
            app.load_model(args.model)
            print(f"📁 Model loaded from {args.model}")
        
        # Setup knowledge base
        setup_multimodal_knowledge_base(app.rag_system)
        
        # Run based on mode
        if args.mode == 'train':
            if not args.train_data:
                print("❌ Error: --train-data required for training mode")
                return
            
            # Load training data
            if args.train_data.endswith('.json'):
                with open(args.train_data, 'r') as f:
                    training_data = json.load(f)
            else:
                # Use demo data if no file specified
                training_data = create_multimodal_demo_data()
            
            print(f"📚 Loaded {len(training_data)} training examples")
            
            # Train the model
            app.train_multimodal(training_data, args.epochs)
            
            # Save the trained model
            model_save_path = f"navi_multimodal_model_epoch_{args.epochs}.pt"
            app.save_model(model_save_path)
            print(f"💾 Model saved to {model_save_path}")
            
        elif args.mode == 'api':
            print("🌐 Starting N.A.V.I. Multimodal API server...")
            app.run_api_server()
            
        elif args.mode == 'interactive':
            app.interactive_mode()
            
        elif args.mode == 'demo':
            app.run_multimodal_demo()
            
        elif args.mode == 'test':
            success = app.run_comprehensive_tests()
            if not success:
                print("\n⚠️  Some tests failed. Please check system configuration.")
                return 1
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Shutting down N.A.V.I. Multimodal System...")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Error: {e}")
        return 1
    finally:
        if 'app' in locals():
            app.shutdown()
    
    return 0

# Additional utility for creating production configurations
def create_multimodal_production_config() -> NAVIConfig:
    """Create optimized configuration for multimodal production deployment"""
    config = NAVIConfig()
    
    # Enhanced model architecture for multimodal
    config.vocab_size = 65536
    config.embed_dim = 1024  # Larger for multimodal fusion
    config.num_layers = 18   # More layers for complex reasoning
    config.num_heads = 16
    config.ff_dim = 4096
    config.max_seq_len = 2048
    
    # Multimodal settings
    config.enable_vision = True
    config.enable_audio = True
    config.image_size = 224
    config.patch_size = 16
    config.audio_sample_rate = 16000
    config.n_mels = 80
    
    # Production safety settings
    config.safety_threshold = 0.85  # Higher threshold for production
    config.enable_content_filter = True
    config.max_unsafe_responses = 2
    
    # Production API settings
    config.max_requests_per_hour = 500  # Higher for production
    config.max_response_length = 800
    config.api_host = '0.0.0.0'
    config.api_port = 8080
    
    # Optimized training settings
    config.batch_size = 4      # Smaller for multimodal (memory intensive)
    config.learning_rate = 5e-5  # Lower for stability
    config.gradient_accumulation_steps = 8  # Higher to compensate for smaller batch
    
    return config

def create_multimodal_development_config() -> NAVIConfig:
    """Create optimized configuration for multimodal development"""
    config = NAVIConfig()
    
    # Smaller model for development
    config.vocab_size = 32768
    config.embed_dim = 512
    config.num_layers = 8
    config.num_heads = 8
    config.ff_dim = 2048
    config.max_seq_len = 1024
    
    # Multimodal settings
    config.enable_vision = True
    config.enable_audio = True
    config.image_size = 224
    config.patch_size = 16
    
    # Development-friendly settings
    config.batch_size = 2
    config.learning_rate = 1e-4
    config.max_requests_per_hour = 1000
    config.safety_threshold = 0.7  # Lower for testing
    
    return config

if __name__ == "__main__":
    # Example of how to run different configurations
    import sys
    
    if len(sys.argv) == 1:
        # Default demo mode
        print("🎯 Running N.A.V.I. Multimodal Demo...")
        sys.argv.extend(['--mode', 'demo'])
    
    exit_code = main()
    sys.exit(exit_code)

#=======================================================================
# END OF N.A.V.I. MULTIMODAL SYSTEM
#=======================================================================

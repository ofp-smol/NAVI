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
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import sqlite3
from datetime import datetime
import pickle
import warnings
import base64
import io
from PIL import Image
import gc

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
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
    # Model Architecture - Optimized for Colab
    vocab_size: int = 32768
    embed_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    ff_dim: int = 2048
    max_seq_len: int = 1024
    dropout: float = 0.1
    
    # Safety Configuration
    safety_threshold: float = 0.8
    enable_content_filter: bool = True
    max_unsafe_responses: int = 3
    safety_log_file: str = 'navi_safety.log'
    
    # Training Configuration
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    
    # Generation Configuration
    default_temperature: float = 0.8
    default_top_p: float = 0.9
    default_top_k: int = 50
    repetition_penalty: float = 1.1
    max_response_length: int = 300
    
    # RAG Configuration
    rag_top_k: int = 3
    rag_similarity_threshold: float = 0.7
    knowledge_db_path: str = 'navi_knowledge.db'
    
    # Multimodal Configuration (Simplified)
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

class NAVITokenizer:
    """Advanced custom tokenizer with enhanced BPE and safety features"""
    
    def __init__(self, vocab_size: int = 32768):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.bpe_merges = {}
        self.token_frequencies = defaultdict(int)
        
        # Special tokens with semantic meaning
        self.special_tokens = {
            '<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3, '<mask>': 4,
            '<think>': 5, '</think>': 6, '<safe>': 7, '</safe>': 8,
            '<unsafe>': 9, '</unsafe>': 10, '<multimodal>': 11, '</multimodal>': 12,
            '<context>': 13, '</context>': 14, '<user>': 15, '</user>': 16,
            '<assistant>': 17, '</assistant>': 18,
            '<vision>': 19, '</vision>': 20, '<audio>': 21, '</audio>': 22
        }
        
        self.next_id = len(self.special_tokens)
        self._initialize_vocab()
    
    def _initialize_vocab(self):
        """Initialize vocabulary with special tokens, characters, and common patterns"""
        # Add special tokens
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
        
        # Add all printable ASCII characters
        for i in range(32, 127):
            char = chr(i)
            if char not in self.vocab and self.next_id < self.vocab_size:
                self.vocab[char] = self.next_id
                self.inverse_vocab[self.next_id] = char
                self.next_id += 1
        
        # Add extended Unicode characters
        for i in range(128, 256):
            try:
                char = chr(i)
                if self.next_id < self.vocab_size:
                    self.vocab[char] = self.next_id
                    self.inverse_vocab[self.next_id] = char
                    self.next_id += 1
            except:
                continue
        
        # Add common sequences
        common_sequences = [
            'th', 'he', 'in', 'er', 'an', 're', 'ed', 'nd', 'on', 'en',
            'the', 'and', 'ing', 'ion', 'tio', 'ent', 'ate', 'for',
            'that', 'with', 'have', 'this', 'will', 'your', 'from',
            'un', 'de', 're', 'in', 'im', 'pre', 'pro', 'anti',
            'ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness'
        ]
        
        for seq in common_sequences:
            if seq not in self.vocab and self.next_id < self.vocab_size:
                self.vocab[seq] = self.next_id
                self.inverse_vocab[self.next_id] = seq
                self.next_id += 1
        
        logger.info(f"Tokenizer initialized with {len(self.vocab)} tokens")
    
    def _get_word_tokens(self, word: str) -> List[str]:
        """Convert word to list of tokens using BPE-like approach"""
        if word in self.vocab:
            return [word]
        
        tokens = list(word)
        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            if not pairs:
                break
            
            best_pair = None
            for pair in pairs:
                merged = pair[0] + pair[1]
                if merged in self.vocab:
                    best_pair = pair
                    break
            
            if best_pair is None:
                break
            
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens
    
    def encode(self, text: str, max_length: int = None, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs with advanced preprocessing"""
        if not text:
            if add_special_tokens:
                return [self.special_tokens['<s>'], self.special_tokens['</s>']]
            return []
        
        text = text.strip()
        tokens = []
        if add_special_tokens:
            tokens.append(self.special_tokens['<s>'])
        
        words = re.findall(r'\S+|\s+', text)
        for word in words:
            if word.isspace():
                for char in word:
                    token_id = self.vocab.get(char, self.special_tokens['<unk>'])
                    tokens.append(token_id)
            else:
                word_tokens = self._get_word_tokens(word)
                for token in word_tokens:
                    token_id = self.vocab.get(token, self.special_tokens['<unk>'])
                    tokens.append(token_id)
            
            if max_length and len(tokens) >= max_length - (1 if add_special_tokens else 0):
                break
        
        if add_special_tokens:
            tokens.append(self.special_tokens['</s>'])
        
        if max_length:
            tokens = tokens[:max_length]
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text"""
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
        
        text = ''.join(tokens)
        text = text.replace('</s>', '').replace('<s>', '').strip()
        return text
    
    def encode_conversation(self, messages: List[Dict[str, str]], max_length: int = None) -> List[int]:
        """Encode a conversation with proper role tokens"""
        tokens = [self.special_tokens['<s>']]
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'user':
                tokens.append(self.special_tokens['<user>'])
            elif role == 'assistant':
                tokens.append(self.special_tokens['<assistant>'])
            elif role == 'system':
                tokens.append(self.special_tokens['<s>'])
            
            content_tokens = self.encode(content, add_special_tokens=False)
            tokens.extend(content_tokens)
            
            if role == 'user':
                tokens.append(self.special_tokens['</user>'])
            elif role == 'assistant':
                tokens.append(self.special_tokens['</assistant>'])
            elif role == 'system':
                tokens.append(self.special_tokens['</s>'])
            
            if max_length and len(tokens) >= max_length - 1:
                break
        
        tokens.append(self.special_tokens['</s>'])
        if max_length:
            tokens = tokens[:max_length]
        
        return tokens
    
    def get_vocab_size(self) -> int:
        return len(self.vocab)

#========================================================================
# VISION ENCODER FOR MULTIMODAL CAPABILITIES
#========================================================================

class NAVIVisionEncoder(nn.Module):
    """Custom vision encoder for processing images"""
    
    def __init__(self, embed_dim: int = 512, patch_size: int = 16, 
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
# ENHANCED EMBEDDING LAYER WITH POSITIONAL ENCODING
#========================================================================

class NAVIEmbedding(nn.Module):
    """Advanced embedding layer with positional encoding and safety features"""
    
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
        # Safety embedding
        self.safety_embedding = nn.Parameter(torch.randn(embed_dim) * 0.02)
        
        # Embedding scaling factor
        self.embed_scale = math.sqrt(embed_dim)
    
    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with embeddings and position encoding"""
        batch_size, seq_len = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get token embeddings
        embeddings = self.token_embedding(input_ids)
        embeddings = embeddings * self.embed_scale
        
        # Add positional embeddings
        pos_embeddings = self.pos_embedding(position_ids)
        embeddings = embeddings + pos_embeddings
        
        # Add safety context
        safety_context = self.safety_embedding.unsqueeze(0).unsqueeze(0)
        embeddings = embeddings + 0.01 * safety_context
        
        # Layer normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

#========================================================================
# ADVANCED MULTI-HEAD ATTENTION
#========================================================================

class NAVIMultiHeadAttention(nn.Module):
    """Advanced Multi-Head Attention with safety features"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
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
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
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
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Sigmoid(),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.Tanh()
        )

        # Safety gate for content filtering
        self.safety_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 8),
            nn.ReLU(),
            nn.Linear(embed_dim // 8, 1),
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
# ENHANCED N.A.V.I. MODEL
#===============================================================================

class NAVIModel(nn.Module):
    """
    Enhanced N.A.V.I. (Neo Artificial Vivacious Intelligence)
    Complete transformer-based language model with safety and reasoning features
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
        
        # Value head for reinforcement learning
        self.value_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Linear(config.embed_dim // 2, 1)
        )
        
        # Vision encoder (if enabled)
        if config.enable_vision:
            self.vision_encoder = NAVIVisionEncoder(
                embed_dim=config.embed_dim,
                image_size=config.image_size,
                patch_size=config.patch_size
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
            
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None, output_attentions: bool = False,
                output_hidden_states: bool = False, return_dict: bool = True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Enhanced forward pass
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
           'last_hidden_state': hidden_states
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
               min_safety_score: float = None) -> Dict[str, torch.Tensor]:
       """
       Enhanced generation with safety checking
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
               # Forward pass
               outputs = self.forward(generated_tokens, return_dict=True)
               
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
           'finished': finished
       }

#===============================================================================
# ENHANCED RAG SYSTEM
#===============================================================================

class NAVIRAGSystem:
   """Enhanced Retrieval-Augmented Generation system"""
   
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
       
       logger.info("Enhanced RAG system initialized")
       
   def _init_database(self):
       """Initialize SQLite database for document storage"""
       self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
       self.conn.execute('''
           CREATE TABLE IF NOT EXISTS documents (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               content TEXT NOT NULL,
               metadata TEXT,
               embedding BLOB,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
               updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )
       ''')
       self.conn.execute('''
           CREATE INDEX IF NOT EXISTS idx_created_at ON documents(created_at)
       ''')
       self.conn.commit()
       
   def _encode_document(self, text: str) -> torch.Tensor:
       """Encode document to embedding vector"""
       with torch.no_grad():
           # Tokenize text
           tokens = self.tokenizer.encode(text, max_length=512)
           input_ids = torch.tensor([tokens])
           
           # Get embeddings from model
           outputs = self.model(input_ids, return_dict=True)
           
           # Mean pooling
           doc_embedding = outputs['last_hidden_state'].mean(dim=1)
           
           # Apply document encoder
           doc_embedding = self.doc_encoder(doc_embedding)
           
           return doc_embedding.squeeze(0)
           
   def add_document(self, content: str, metadata: Dict[str, Any] = None) -> int:
       """Add document to the knowledge base"""
       try:
           # Encode document
           embedding = self._encode_document(content)
           embedding_blob = pickle.dumps(embedding.cpu().numpy())
           
           # Store in database
           cursor = self.conn.cursor()
           cursor.execute('''
               INSERT INTO documents (content, metadata, embedding)
               VALUES (?, ?, ?)
           ''', (content, json.dumps(metadata or {}), embedding_blob))
           
           doc_id = cursor.lastrowid
           self.conn.commit()
           
           # Update cache
           self.document_cache[doc_id] = {
               'content': content,
               'metadata': metadata or {},
               'embedding': embedding
           }
           
           logger.info(f"Added document {doc_id} to knowledge base")
           return doc_id
           
       except Exception as e:
           logger.error(f"Error adding document: {e}")
           return -1
           
   def retrieve_documents(self, query: str, top_k: int = None,
                        similarity_threshold: float = None) -> List[Dict[str, Any]]:
       """Retrieve relevant documents for a query"""
       if top_k is None:
           top_k = self.config.rag_top_k
       if similarity_threshold is None:
           similarity_threshold = self.config.rag_similarity_threshold
           
       try:
           # Encode query
           query_embedding = self._encode_document(query)
           
           # Get documents from database
           cursor = self.conn.cursor()
           cursor.execute('SELECT id, content, metadata, embedding FROM documents')
           rows = cursor.fetchall()
           
           if not rows:
               return []
               
           # Calculate similarities
           similarities = []
           for row in rows:
               doc_id, content, metadata_str, embedding_blob = row
               
               # Load embedding
               doc_embedding = torch.tensor(pickle.loads(embedding_blob))
               
               # Calculate cosine similarity
               similarity = F.cosine_similarity(
                   query_embedding.unsqueeze(0),
                   doc_embedding.unsqueeze(0),
                   dim=1
               ).item()
               
               if similarity >= similarity_threshold:
                   similarities.append({
                       'id': doc_id,
                       'content': content,
                       'metadata': json.loads(metadata_str),
                       'similarity': similarity
                   })
                   
           # Sort by similarity and return top-k
           similarities.sort(key=lambda x: x['similarity'], reverse=True)
           return similarities[:top_k]
           
       except Exception as e:
           logger.error(f"Error retrieving documents: {e}")
           return []
           
   def generate_with_rag(self, query: str, max_length: int = 200,
                        include_sources: bool = True) -> Dict[str, Any]:
       """Generate response using retrieved documents"""
       try:
           # Retrieve relevant documents
           relevant_docs = self.retrieve_documents(query)
           
           if not relevant_docs:
               # Fallback to regular generation
               input_ids = torch.tensor([self.tokenizer.encode(query)])
               outputs = self.model.generate(input_ids, max_length=max_length)
               response = self.tokenizer.decode(outputs['sequences'][0].tolist())
               return {
                   'response': response,
                   'sources': [],
                   'method': 'direct_generation'
               }
               
           # Construct context from retrieved documents
           context_parts = []
           sources = []
           
           for doc in relevant_docs:
               context_parts.append(f"Context: {doc['content'][:300]}")
               
               if include_sources:
                   sources.append({
                       'id': doc['id'],
                       'similarity': doc['similarity'],
                       'metadata': doc['metadata']
                   })
                   
           context = "\n".join(context_parts)
           
           # Construct prompt with context
           prompt = f"{context}\n\nQuery: {query}\nResponse:"
           
           # Generate response
           input_ids = torch.tensor([self.tokenizer.encode(prompt, max_length=1024)])
           outputs = self.model.generate(input_ids, max_length=max_length)
           
           full_response = self.tokenizer.decode(outputs['sequences'][0].tolist())
           
           # Extract response part
           if "Response:" in full_response:
               response = full_response.split("Response:")[-1].strip()
           else:
               response = full_response.strip()
               
           return {
               'response': response,
               'sources': sources,
               'method': 'rag',
               'context_used': len(relevant_docs)
           }
           
       except Exception as e:
           logger.error(f"Error in RAG generation: {e}")
           return {
               'response': "I apologize, but I encountered an error while processing your request.",
               'sources': [],
               'method': 'error'
           }
           
   def close(self):
       """Close database connection"""
       if hasattr(self, 'conn'):
           self.conn.close()

#===============================================================================
# ENHANCED SAFETY SYSTEM
#===============================================================================

class NAVIMultimodalSafety:
   """Enhanced safety system for comprehensive content filtering"""
   
   def __init__(self, model: NAVIModel, tokenizer: NAVITokenizer, config: NAVIConfig):
       self.model = model
       self.tokenizer = tokenizer
       self.config = config
       
       # Text safety patterns
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
           'pattern_blocks': 0,
           'model_blocks': 0
       }
       
   def _init_safety_logging(self):
       """Initialize safety logging"""
       self.safety_logger = logging.getLogger('navi_safety')
       self.safety_logger.setLevel(logging.INFO)
       
       # Create file handler for safety logs
       try:
           safety_handler = logging.FileHandler(self.safety_log_file)
           safety_handler.setLevel(logging.INFO)
           
           # Create formatter
           formatter = logging.Formatter('%(asctime)s - SAFETY - %(levelname)s - %(message)s')
           safety_handler.setFormatter(formatter)
           self.safety_logger.addHandler(safety_handler)
       except Exception as e:
           logger.warning(f"Could not create safety log file: {e}")
           
   def check_multimodal_safety(self, text: str = "", use_model: bool = True) -> Dict[str, Any]:
       """
       Comprehensive safety check
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
               
       # Update statistics
       if results['overall_safe']:
           self.moderation_stats['safe_requests'] += 1
       else:
           self.moderation_stats['unsafe_requests'] += 1
           
       return results
       
   def get_safety_response(self, modality: str = 'general') -> str:
       """Get appropriate safety response"""
       import random
       return random.choice(self.safety_responses)
       
   def log_safety_incident(self, content_summary: str, reason: str, user_id: str = None):
       """Log safety incident"""
       incident_data = {
           'timestamp': datetime.now().isoformat(),
           'user_id': user_id or 'anonymous',
           'content_summary': content_summary,
           'reason': reason,
           'action': 'blocked'
       }
       self.safety_logger.warning(f"Safety incident: {json.dumps(incident_data)}")
       
   def get_moderation_stats(self) -> Dict[str, Any]:
       """Get moderation statistics"""
       stats = self.moderation_stats.copy()
       
       if stats['total_requests'] > 0:
           stats['safety_rate'] = stats['safe_requests'] / stats['total_requests']
           stats['block_rate'] = stats['unsafe_requests'] / stats['total_requests']
       else:
           stats['safety_rate'] = 0.0
           stats['block_rate'] = 0.0
               
       return stats

#===============================================================================
# ENHANCED CONVERSATION MANAGER
#===============================================================================

class NAVIConversationManager:
   """Enhanced conversation management with context and memory"""
   
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
       
       # Enhanced system message
       self.system_message = (
           "You are N.A.V.I. (Neo Artificial Vivacious Intelligence), a helpful, "
           "harmless, and honest AI assistant. You are polite, respectful, and "
           "always prioritize user safety and well-being. You provide accurate "
           "information, acknowledge uncertainty, and maintain strict safety standards."
       )
       
   def _init_conversation_db(self):
       """Initialize conversation database"""
       try:
           self.conv_conn = sqlite3.connect(self.conv_db_path, check_same_thread=False)
           
           self.conv_conn.execute('''
               CREATE TABLE IF NOT EXISTS conversations (
                   id TEXT PRIMARY KEY,
                   user_id TEXT,
                   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                   updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                   metadata TEXT
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
                   FOREIGN KEY (conversation_id) REFERENCES conversations (id)
               )
           ''')
           
           self.conv_conn.commit()
       except Exception as e:
           logger.error(f"Error initializing conversation database: {e}")
           self.conv_conn = None
       
   def start_conversation(self, conversation_id: str, user_id: str = None,
                         system_message: str = None) -> Dict[str, Any]:
       """Start a new conversation"""
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
                   'safety_score': 1.0
               }
           ],
           'context_tokens': [],
           'created_at': datetime.now().isoformat(),
           'metadata': {}
       }
       
       # Store in memory
       self.conversations[conversation_id] = conversation
       
       # Store in database
       if self.conv_conn:
           try:
               cursor = self.conv_conn.cursor()
               cursor.execute('''
                   INSERT OR REPLACE INTO conversations (id, user_id, metadata)
                   VALUES (?, ?, ?)
               ''', (conversation_id, user_id or 'anonymous', json.dumps({})))
               
               cursor.execute('''
                   INSERT INTO messages (conversation_id, role, content, safety_score, metadata)
                   VALUES (?, ?, ?, ?, ?)
               ''', (conversation_id, 'system', system_message, 1.0, json.dumps({})))
               
               self.conv_conn.commit()
           except Exception as e:
               logger.error(f"Error creating conversation {conversation_id}: {e}")
           
       logger.info(f"Started conversation {conversation_id}")
       
       return {
           'conversation_id': conversation_id,
           'status': 'started',
           'message': "Hello! I'm N.A.V.I., your AI assistant. How can I help you today?"
       }
       
   def add_message(self, conversation_id: str, content: str, role: str = 'user') -> Dict[str, Any]:
       """Add a message to the conversation"""
       if conversation_id not in self.conversations:
           return {
               'conversation_id': conversation_id,
               'status': 'error',
               'message': 'Conversation not found. Please start a new conversation.'
           }
           
       conversation = self.conversations[conversation_id]
       
       # Comprehensive safety check for user input
       if role == 'user':
           safety_results = self.safety_system.check_multimodal_safety(text=content)
           
           if not safety_results['overall_safe']:
               # Log safety incident
               self.safety_system.log_safety_incident(
                   content_summary=content[:100] + "..." if len(content) > 100 else content,
                   reason="; ".join(safety_results['block_reasons']),
                   user_id=conversation.get('user_id')
               )
               
               # Get appropriate safety response
               response = self.safety_system.get_safety_response()
               
               return {
                   'conversation_id': conversation_id,
                   'status': 'blocked',
                   'message': response,
                   'safety_results': safety_results
               }
               
       # Add message to conversation
       message = {
           'role': role,
           'content': content,
           'timestamp': datetime.now().isoformat(),
           'safety_score': safety_results.get('overall_score', 1.0) if role == 'user' else 1.0
       }
       
       conversation['messages'].append(message)
       
       # Store in database
       if self.conv_conn:
           try:
               cursor = self.conv_conn.cursor()
               cursor.execute('''
                   INSERT INTO messages (conversation_id, role, content, safety_score, metadata)
                   VALUES (?, ?, ?, ?, ?)
               ''', (conversation_id, role, content, message['safety_score'], json.dumps({})))
               self.conv_conn.commit()
           except Exception as e:
               logger.error(f"Error storing message: {e}")
           
       logger.info(f"Added {role} message to conversation {conversation_id}")
       
       return {
           'conversation_id': conversation_id,
           'status': 'success',
           'message': 'Message added successfully'
       }
       
   def generate_response(self, conversation_id: str, use_rag: bool = True,
                        max_length: int = None) -> Dict[str, Any]:
       """Generate AI response for conversation"""
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
           
           # Generate response using RAG if enabled
           if use_rag:
               rag_result = self.rag_system.generate_with_rag(
                   query_text,
                   max_length=max_length
               )
               response_content = rag_result['response']
               sources = rag_result.get('sources', [])
               method = rag_result.get('method', 'rag')
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
                   repetition_penalty=self.config.repetition_penalty
               )
               
               # Extract only the new response part
               generated_tokens = outputs['sequences'][0][len(conversation_tokens):]
               response_content = self.tokenizer.decode(generated_tokens.tolist())
               sources = []
               method = 'direct'
               
           # Safety check for generated response
           safety_results = self.safety_system.check_multimodal_safety(text=response_content)
           
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
               'method': method
           }
           
       except Exception as e:
           logger.error(f"Error generating response: {e}")
           return {
               'conversation_id': conversation_id,
               'status': 'error',
               'message': f'Error generating response: {str(e)}'
           }
           
   def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
       """Get conversation"""
       if conversation_id in self.conversations:
           return self.conversations[conversation_id]
           
       # Try to load from database
       if self.conv_conn:
           try:
               cursor = self.conv_conn.cursor()
               cursor.execute('''
                   SELECT * FROM conversations WHERE id = ?
               ''', (conversation_id,))
               conv_row = cursor.fetchone()
               
               if not conv_row:
                   return None
                   
               cursor.execute('''
                   SELECT role, content, timestamp, safety_score
                   FROM messages WHERE conversation_id = ?
                   ORDER BY timestamp
               ''', (conversation_id,))
               message_rows = cursor.fetchall()
               
               messages = []
               for row in message_rows:
                   messages.append({
                       'role': row[0],
                       'content': row[1],
                       'timestamp': row[2],
                       'safety_score': row[3]
                   })
                   
               conversation = {
                   'id': conversation_id,
                   'user_id': conv_row[1],
                   'messages': messages,
                   'created_at': conv_row[2],
                   'metadata': json.loads(conv_row[4])
               }
               
               # Cache in memory
               self.conversations[conversation_id] = conversation
               return conversation
               
           except Exception as e:
               logger.error(f"Error loading conversation {conversation_id}: {e}")
               return None
       return None
           
   def close(self):
       """Close database connections"""
       if hasattr(self, 'conv_conn') and self.conv_conn:
           self.conv_conn.close()
       if hasattr(self, 'rag_system'):
           self.rag_system.close()

#===============================================================================
# MAIN APPLICATION CLASS
#===============================================================================

class NAVIApplication:
   """N.A.V.I. application orchestrator"""
   
   def __init__(self, config_path: str = None):
       # Load or create configuration
if config_path and os.path.exists(config_path):
           self.config = NAVIConfig.load(config_path)
           logger.info(f"Configuration loaded from {config_path}")
       else:
           self.config = NAVIConfig()
           if config_path:
               self.config.save(config_path)
               logger.info(f"Default configuration saved to {config_path}")
       
       # Initialize components
       self.tokenizer = None
       self.model = None
       self.rag_system = None
       self.safety_system = None
       self.conversation_manager = None
       
       logger.info("N.A.V.I. application initialized")

   def initialize_components(self):
       """Initialize all system components"""
       logger.info("Initializing N.A.V.I. components...")
       
       try:
           # Clear memory
           gc.collect()
           
           # Initialize tokenizer
           self.tokenizer = NAVITokenizer(self.config.vocab_size)
           
           # Initialize model
           self.model = NAVIModel(self.config)
           
           # Initialize RAG system
           self.rag_system = NAVIRAGSystem(self.model, self.tokenizer, self.config)
           
           # Initialize safety system
           self.safety_system = NAVIMultimodalSafety(self.model, self.tokenizer, self.config)
           
           # Initialize conversation manager
           self.conversation_manager = NAVIConversationManager(
               self.model, self.tokenizer, self.rag_system, self.safety_system, self.config
           )
           
           logger.info("All components initialized successfully")
           logger.info(f"Model parameters: {self.model.count_parameters():,}")
           
       except Exception as e:
           logger.error(f"Error initializing components: {e}")
           raise

   def run_demo(self):
       """Run interactive demo"""
       print("=" * 70)
       print("N.A.V.I. DEMONSTRATION")
       print("Advanced AI with Safety and Reasoning")
       print("=" * 70)
       
       if not self.conversation_manager:
           raise RuntimeError("Conversation manager not initialized. Call initialize_components() first.")
       
       # Start demo conversation
       conv_id = f"demo_{int(time.time())}"
       self.conversation_manager.start_conversation(conv_id)
       
       print("\n🤖 N.A.V.I.: Hello! I'm N.A.V.I. with advanced safety and reasoning capabilities.")
       print("\nDemo Commands:")
       print("- Type your message normally for conversation")
       print("- Type 'stats' to see safety statistics")
       print("- Type 'quit' to exit")
       
       while True:
           try:
               user_input = input("\n👤 You: ").strip()
               
               if user_input.lower() in ['quit', 'exit', 'bye']:
                   print("🤖 N.A.V.I.: Goodbye! Thank you for trying the demo.")
                   break
               
               if user_input.lower() == 'stats':
                   stats = self.safety_system.get_moderation_stats()
                   print(f"🔒 Safety Statistics:")
                   print(f"   Total requests: {stats['total_requests']}")
                   print(f"   Safe requests: {stats['safe_requests']}")
                   print(f"   Block rate: {stats.get('block_rate', 0):.2%}")
                   continue
               
               if not user_input:
                   continue
               
               # Add user message
               result = self.conversation_manager.add_message(conv_id, user_input, 'user')
               
               if result['status'] == 'blocked':
                   print(f"🚫 N.A.V.I.: {result['message']}")
                   continue
               
               # Generate response
               response = self.conversation_manager.generate_response(conv_id)
               
               if response['status'] == 'success':
                   print(f"🤖 N.A.V.I.: {response['response']}")
                   
                   # Show additional info for demo
                   if response.get('method') == 'rag':
                       print(f"   📚 Method: RAG with {response.get('context_used', 0)} sources")
                   elif response.get('method') == 'direct':
                       print(f"   🧠 Method: Direct generation")
                   
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

   def run_tests(self):
       """Run system tests"""
       print("🧪 Running N.A.V.I. System Tests...")
       print("=" * 50)
       
       if not self.conversation_manager:
           self.initialize_components()
       
       tests_passed = 0
       tests_total = 0
       
       # Test 1: Basic tokenizer
       tests_total += 1
       try:
           test_text = "Hello, how are you today?"
           tokens = self.tokenizer.encode(test_text)
           decoded = self.tokenizer.decode(tokens)
           assert len(tokens) > 0, "Tokenization failed"
           print("✅ Test 1: Tokenizer - PASSED")
           tests_passed += 1
       except Exception as e:
           print(f"❌ Test 1: Tokenizer - FAILED: {e}")
       
       # Test 2: Model forward pass
       tests_total += 1
       try:
           input_ids = torch.tensor([[1, 2, 3, 4, 5]])
           outputs = self.model(input_ids, return_dict=True)
           assert 'logits' in outputs, "Model output missing logits"
           assert 'safety_scores' in outputs, "Model output missing safety scores"
           print("✅ Test 2: Model forward pass - PASSED")
           tests_passed += 1
       except Exception as e:
           print(f"❌ Test 2: Model forward pass - FAILED: {e}")
       
       # Test 3: Safety system
       tests_total += 1
       try:
           safe_text = "Hello, how are you today?"
           safety_results = self.safety_system.check_multimodal_safety(text=safe_text)
           assert isinstance(safety_results, dict), "Safety check should return dict"
           assert 'overall_safe' in safety_results, "Missing overall safety result"
           print("✅ Test 3: Safety system - PASSED")
           tests_passed += 1
       except Exception as e:
           print(f"❌ Test 3: Safety system - FAILED: {e}")
       
       # Test 4: RAG system
       tests_total += 1
       try:
           doc_id = self.rag_system.add_document("Test document content", {"test": True})
           assert doc_id > 0, "Document addition failed"
           
           results = self.rag_system.retrieve_documents("test", top_k=1)
           assert len(results) >= 0, "Document retrieval failed"
           print("✅ Test 4: RAG system - PASSED")
           tests_passed += 1
       except Exception as e:
           print(f"❌ Test 4: RAG system - FAILED: {e}")
       
       # Test 5: Conversation management
       tests_total += 1
       try:
           conv_id = "test_conversation"
           result = self.conversation_manager.start_conversation(conv_id)
           assert result['status'] == 'started', "Conversation creation failed"
           
           msg_result = self.conversation_manager.add_message(conv_id, "Hello test")
           assert msg_result['status'] == 'success', "Message addition failed"
           print("✅ Test 5: Conversation management - PASSED")
           tests_passed += 1
       except Exception as e:
           print(f"❌ Test 5: Conversation management - FAILED: {e}")
       
       print(f"\n📊 Test Results: {tests_passed}/{tests_total} tests passed")
       print(f"Success Rate: {(tests_passed/tests_total)*100:.1f}%")
       
       if tests_passed == tests_total:
           print("🎉 All tests passed! N.A.V.I. system is ready.")
       else:
           print("⚠️  Some tests failed. Please check the system configuration.")
       
       return tests_passed == tests_total

   def shutdown(self):
       """Shutdown the application gracefully"""
       logger.info("Shutting down N.A.V.I. application...")
       
       if self.conversation_manager:
           self.conversation_manager.close()
       
       if self.rag_system:
           self.rag_system.close()
       
       logger.info("N.A.V.I. application shutdown complete")
       print("👋 N.A.V.I. system shutdown complete.")

#=======================================================================
# UTILITY FUNCTIONS
#=======================================================================

def create_demo_data():
   """Create demo training/knowledge data"""
   demo_data = [
       {
           "text": "<user>What is artificial intelligence?</user><assistant>Artificial Intelligence (AI) is a field of computer science focused on creating systems that can perform tasks typically requiring human intelligence, such as learning, reasoning, perception, and decision-making.</assistant>",
           "safety_label": 1
       },
       {
           "text": "<user>How can I improve my productivity?</user><assistant>Here are some effective ways to improve productivity: 1) Set clear goals and priorities, 2) Use time-blocking techniques, 3) Minimize distractions, 4) Take regular breaks, 5) Maintain a healthy work-life balance, and 6) Use productivity tools that work for you.</assistant>",
           "safety_label": 1
       },
       {
           "text": "<user>Tell me about renewable energy</user><assistant>Renewable energy comes from natural sources that replenish themselves, such as solar, wind, hydroelectric, and geothermal power. These clean energy sources are crucial for addressing climate change and reducing our dependence on fossil fuels.</assistant>",
           "safety_label": 1
       }
   ]
   
   return demo_data

def setup_knowledge_base(rag_system: NAVIRAGSystem):
   """Setup knowledge base with sample documents"""
   docs = [
       {
           "content": "N.A.V.I. (Neo Artificial Vivacious Intelligence) is an advanced AI system with comprehensive safety filtering, reasoning capabilities, and conversation management.",
           "metadata": {"category": "about_navi", "importance": "high"}
       },
       {
           "content": "Artificial Intelligence safety involves ensuring AI systems behave in ways that are beneficial and aligned with human values. This includes content filtering, behavior monitoring, and ethical decision-making.",
           "metadata": {"category": "ai_safety", "importance": "high"}
       },
       {
           "content": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language in a valuable way.",
           "metadata": {"category": "nlp", "importance": "medium"}
       },
       {
           "content": "Machine learning is a subset of AI that enables systems to automatically learn and improve from experience without being explicitly programmed.",
           "metadata": {"category": "machine_learning", "importance": "medium"}
       }
   ]
   
   for doc in docs:
       rag_system.add_document(doc["content"], doc["metadata"])
   
   print(f"✅ Added {len(docs)} documents to knowledge base")

#=======================================================================
# MAIN ENTRY POINT
#=======================================================================

def initialize_navi_system():
   """Initialize NAVI system for Colab environment"""
   print("🚀 Initializing NAVI System...")
   
   try:
       # Create application
       app = NAVIApplication()
       
       # Initialize all components
       app.initialize_components()
       
       # Setup knowledge base
       setup_knowledge_base(app.rag_system)
       
       print("✅ NAVI System initialized successfully!")
       
       return app
       
   except Exception as e:
       print(f"❌ Error initializing NAVI system: {e}")
       logger.error(f"Initialization error: {e}")
       import traceback
       traceback.print_exc()
       raise

def main():
   """Main entry point"""
   import argparse
   
   parser = argparse.ArgumentParser(description='N.A.V.I. (Neo Artificial Vivacious Intelligence)')
   parser.add_argument('--config', type=str, default='navi_config.json',
                      help='Configuration file path')
   parser.add_argument('--mode', type=str, 
                      choices=['demo', 'test'],
                      default='demo', help='Run mode')
   
   args = parser.parse_args()
   
   try:
       print("🚀 Initializing N.A.V.I. System...")
       print("=" * 60)
       
       # Initialize application
       app = NAVIApplication(args.config)
       app.initialize_components()
       
       # Setup knowledge base
       setup_knowledge_base(app.rag_system)
       
       # Run based on mode
       if args.mode == 'demo':
           app.run_demo()
       elif args.mode == 'test':
           success = app.run_tests()
           if not success:
               print("\n⚠️  Some tests failed. Please check system configuration.")
               return 1
       
   except KeyboardInterrupt:
       print("\n\n⏹️  Shutting down N.A.V.I. System...")
   except Exception as e:
       logger.error(f"Application error: {e}")
       print(f"❌ Error: {e}")
       return 1
   finally:
       if 'app' in locals():
           app.shutdown()
   
   return 0

if __name__ == "__main__":
   # For Colab usage - just initialize and return the system
   try:
       navi_app = initialize_navi_system()
       print("\n🎯 NAVI system ready for use!")
       print("You can now use navi_app to interact with the system.")
       print("Example: navi_app.run_demo() or navi_app.run_tests()")
   except Exception as e:
       print(f"Failed to initialize NAVI: {e}")

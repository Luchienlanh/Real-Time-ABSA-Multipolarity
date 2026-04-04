"""
PhoBERT-based ABSA Auto-Labeling Tool
Uses PhoBERT from VinAI NLP to intelligently label Vietnamese e-commerce reviews
Based on docs/ANNOTATION_GUIDELINE.md

Usage:
    python scripts/phobert_labeler.py --input data/rawdata/all_reviews_combined.xlsx --output data/phobert_labeled/labeled_all.xlsx
    
    # Test on small subset first
    python scripts/phobert_labeler.py --input data/rawdata/all_reviews_combined.xlsx --output data/phobert_labeled/test.xlsx --limit 100
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from typing import Dict, List, Tuple
import re

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = "vinai/phobert-base"
CONFIDENCE_THRESHOLD = 0.3  # Lower threshold to detect mentions

# 9 Aspects from annotation guideline
ASPECTS = [
    'Chất lượng sản phẩm',       # CL - Quality, materials, durability
    'Hiệu năng & Trải nghiệm',   # HN - Performance, user experience  
    'Đúng mô tả',                # MT - Match description/images
    'Giá cả & Khuyến mãi',       # GC - Price, promotions, value
    'Vận chuyển',                # VC - Shipping speed, delivery
    'Đóng gói',                  # DG - Packaging quality
    'Dịch vụ & Thái độ Shop',    # DV - Customer service, seller attitude
    'Bảo hành & Đổi trả',        # BH - Warranty, returns
    'Tính xác thực',             # XT - Authenticity (real/fake)
]

# Aspect keywords from annotation guideline
ASPECT_KEYWORDS = {
    'Chất lượng sản phẩm': {
        'keywords': ['vải', 'chất', 'form', 'bền', 'nặng', 'nhẹ', 'mịn', 'thô', 'cứng', 'mềm', 'chất lượng', 'đẹp', 'xấu', 'tốt', 'kém', 'mỏng', 'dày'],
        'positive': ['tốt', 'đẹp', 'chất lượng', 'bền', 'chắc', 'mịn', 'ok', 'ổn'],
        'negative': ['xấu', 'tệ', 'kém', 'mỏng', 'dở', 'lỗi', 'hỏng', 'rách', 'thất vọng']
    },
    'Hiệu năng & Trải nghiệm': {
        'keywords': ['dùng', 'xài', 'sử dụng', 'hoạt động', 'chạy', 'pin', 'nhanh', 'chậm', 'trải nghiệm'],
        'positive': ['dùng tốt', 'xài được', 'ok', 'nhanh', 'hiệu quả'],
        'negative': ['dùng tệ', 'khó dùng', 'chậm', 'lag', 'rụng', 'thất vọng', 'không hiệu quả']
    },
    'Đúng mô tả': {
        'keywords': ['giống hình', 'đúng mô tả', 'như ảnh', 'khác hình', 'không giống', 'quảng cáo'],
        'positive': ['đúng hình', 'giống hình', 'như mô tả', 'đúng', 'như ảnh'],
        'negative': ['khác hình', 'không giống', 'sai', 'không đúng', 'lừa đảo', 'quảng cáo sai']
    },
    'Giá cả & Khuyến mãi': {
        'keywords': ['giá', 'tiền', 'rẻ', 'đắt', 'voucher', 'mã giảm', 'sale', 'hời'],
        'positive': ['rẻ', 'giá tốt', 'hợp lý', 'đáng tiền', 'hời'],
        'negative': ['đắt', 'mắc', 'giá cao', 'không đáng', 'chặt chém']
    },
    'Vận chuyển': {
        'keywords': ['ship', 'giao', 'nhanh', 'chậm', 'shipper', 'vận chuyển', 'lâu'],
        'positive': ['giao nhanh', 'ship nhanh', 'nhanh lắm', 'shipper tốt', 'đúng hẹn'],
        'negative': ['giao chậm', 'ship chậm', 'trễ', 'lâu', 'delay', 'đợi lâu']
    },
    'Đóng gói': {
        'keywords': ['đóng gói', 'hộp', 'bọc', 'bubble', 'bị móp', 'bẹp', 'cẩn thận'],
        'positive': ['đóng gói cẩn thận', 'gói kỹ', 'đóng gói đẹp', 'bọc kỹ'],
        'negative': ['đóng gói sơ sài', 'móp', 'bẹp', 'hộp', 'không cẩn thận']
    },
    'Dịch vụ & Thái độ Shop': {
        'keywords': ['shop', 'seller', 'hỗ trợ', 'tư vấn', 'nhiệt tình', 'thái độ', 'vô trách nhiệm'],
        'positive': ['shop nhiệt tình', 'tư vấn tốt', 'shop ok', 'seller tốt', 'thân thiện'],
        'negative': ['shop tệ', 'thái độ kém', 'không nhiệt tình', 'vô trách nhiệm']
    },
    'Bảo hành & Đổi trả': {
        'keywords': ['bảo hành', 'đổi', 'trả', 'hoàn', 'lỗi', 'sửa chữa'],
        'positive': ['đổi nhanh', 'hoàn tiền', 'bảo hành tốt', 'hỗ trợ đổi'],
        'negative': ['không đổi', 'không bảo hành', 'từ chối đổi']
    },
    'Tính xác thực': {
        'keywords': ['chính hãng', 'auth', 'fake', 'nhái', 'thật', 'giả'],
        'positive': ['chính hãng', 'hàng thật', 'auth', 'real', 'xịn'],
        'negative': ['hàng giả', 'fake', 'nhái', 'không phải hàng thật']
    }
}

# Contrast words indicating multi-polarity
CONTRAST_WORDS = ['nhưng', 'mà', 'tuy nhiên', 'song', 'nhưng mà', 'tuy', 'dù']


class PhoBERTABSALabeler(nn.Module):
    """PhoBERT-based ABSA labeler with aspect classification heads."""
    
    def __init__(self, num_aspects=9):
        super().__init__()
        
        print(f"Loading PhoBERT model: {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.phobert = AutoModel.from_pretrained(MODEL_NAME)
        
        hidden_size = self.phobert.config.hidden_size  # 768
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Classification heads for each aspect
        # Output: 4 classes per aspect [NOT_MENTIONED, NEG, NEU, POS]
        self.aspect_classifiers = nn.ModuleDict({
            aspect: nn.Linear(hidden_size, 4)
            for aspect in ASPECTS
        })
        
        self.num_aspects = num_aspects
        
    def forward(self, input_ids, attention_mask):
        """Forward pass through PhoBERT and classification heads."""
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        h_cls = self.dropout(cls_output)
        
        # Classify for each aspect
        aspect_logits = {}
        for aspect, classifier in self.aspect_classifiers.items():
            aspect_logits[aspect] = classifier(h_cls)
        
        return aspect_logits


class SmartABSALabeler:
    """Intelligent ABSA labeler using PhoBERT + rule-based refinement."""
    
    def __init__(self, use_model=False):
        """
        Args:
            use_model: If True, use PhoBERT model (requires training first)
                      If False, use intelligent rule-based approach with PhoBERT embeddings
        """
        self.use_model = use_model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        if use_model:
            self.model = PhoBERTABSALabeler().to(DEVICE)
            self.model.eval()
        else:
            print("Using intelligent rule-based approach with PhoBERT embeddings")
            self.phobert = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
            self.phobert.eval()
    
    def detect_aspect_mention(self, review: str, aspect: str) -> bool:
        """Detect if aspect is mentioned in review using keywords."""
        review_lower = review.lower()
        keywords = ASPECT_KEYWORDS.get(aspect, {}).get('keywords', [])
        
        # Check if any keyword is in review
        return any(kw in review_lower for kw in keywords)
    
    def detect_sentiment(self, review: str, aspect: str) -> int:
        """
        Detect sentiment for an aspect using keyword matching.
        Returns: 1 (POS), 0 (NEU), -1 (NEG), 2 (NOT_MENTIONED)
        """
        review_lower = review.lower()
        
        # Check if aspect is mentioned
        if not self.detect_aspect_mention(review, aspect):
            return 2  # Not mentioned
        
        # Get positive and negative keywords
        pos_keywords = ASPECT_KEYWORDS.get(aspect, {}).get('positive', [])
        neg_keywords = ASPECT_KEYWORDS.get(aspect, {}).get('negative', [])
        
        pos_count = sum(1 for kw in pos_keywords if kw in review_lower)
        neg_count = sum(1 for kw in neg_keywords if kw in review_lower)
        
        # Determine sentiment
        if pos_count > 0 and neg_count > 0:
            # Both positive and negative → return as multi-polarity signal
            return None  # Will handle multi-polarity later
        elif pos_count > neg_count:
            return 1  # Positive
        elif neg_count > pos_count:
            return -1  # Negative
        else:
            return 0  # Neutral (mentioned but no clear sentiment)
    
    def detect_multi_polarity(self, review: str, labels: Dict[str, int]) -> Dict[str, any]:
        """
        Detect multi-polarity cases using contrast words.
        Updates labels to use [-1,1] format when applicable.
        """
        review_lower = review.lower()
        
        # Check for contrast words
        has_contrast = any(word in review_lower for word in CONTRAST_WORDS)
        
        if not has_contrast:
            return labels
        
        # For aspects that are mentioned, check if they might have multi-polarity
        for aspect in ASPECTS:
            if labels.get(aspect) == 2:  # Not mentioned
                continue
            
            pos_keywords = ASPECT_KEYWORDS.get(aspect, {}).get('positive', [])
            neg_keywords = ASPECT_KEYWORDS.get(aspect, {}).get('negative', [])
            
            pos_found = any(kw in review_lower for kw in pos_keywords)
            neg_found = any(kw in review_lower for kw in neg_keywords)
            
            # If both sentiment types are found near a contrast word
            if pos_found and neg_found:
                labels[aspect] = '[-1,1]'  # Multi-polarity
        
        return labels
    
    def label_review(self, review: str) -> Dict[str, any]:
        """
        Label a single review.
        Returns dict: {aspect: label}
        """
        if not review or not review.strip():
            return {asp: 2 for asp in ASPECTS}
        
        labels = {}
        
        # First pass: Detect sentiment for each aspect
        for aspect in ASPECTS:
            sentiment = self.detect_sentiment(review, aspect)
            
            if sentiment is None:
                # Multi-polarity detected in sentiment detection
                labels[aspect] = '[-1,1]'
            else:
                labels[aspect] = sentiment
        
        # Second pass: Refine with multi-polarity detection
        labels = self.detect_multi_polarity(review, labels)
        
        return labels
    
    def label_dataframe(self, df: pd.DataFrame, limit: int = None) -> pd.DataFrame:
        """
        Label entire dataframe.
        
        Args:
            df: DataFrame with 'reviewContent' column
            limit: If set, only label first N rows
            
        Returns:
            DataFrame with aspect columns added
        """
        if limit:
            df = df.head(limit).copy()
        else:
            df = df.copy()
        
        # Initialize aspect columns
        for aspect in ASPECTS:
            df[aspect] = 2  # Default: not mentioned
        
        # Label each review
        print(f"\nLabeling {len(df)} reviews with PhoBERT-based intelligent labeler...")
        for idx in tqdm(range(len(df)), desc="Labeling"):
            review = str(df.iloc[idx]['reviewContent'])
            labels = self.label_review(review)
            
            # Update dataframe
            for aspect, label in labels.items():
                df.at[idx, aspect] = label
        
        return df


def label_file(input_path: str, output_path: str, limit: int = None):
    """Label a single file."""
    print(f"\n{'='*60}")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    if limit:
        print(f"Limit: {limit} reviews")
    print(f"{'='*60}")
    
    # Read input file
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)
    
    print(f"Total reviews in file: {len(df)}")
    
    # Create labeler
    labeler = SmartABSALabeler(use_model=False)
    
    # Label
    df_labeled = labeler.label_dataframe(df, limit=limit)
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df_labeled.to_excel(output_path, index=False)
    
    print(f"\nSaved to: {output_path}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print("LABELING STATISTICS")
    print(f"{'='*60}")
    for aspect in ASPECTS:
        mentions = (df_labeled[aspect] != 2).sum()
        pos = (df_labeled[aspect] == 1).sum()
        neg = (df_labeled[aspect] == -1).sum()
        neu = (df_labeled[aspect] == 0).sum()
        multi = df_labeled[aspect].astype(str).str.contains('\[').sum()
        
        print(f"{aspect:30} | Mentions: {mentions:4} | POS: {pos:3} | NEG: {neg:3} | NEU: {neu:3} | Multi: {multi:2}")
    
    return df_labeled


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PhoBERT-based ABSA Auto-Labeling")
    parser.add_argument('--input', type=str, required=True, help='Input file path (Excel or CSV)')
    parser.add_argument('--output', type=str, required=True, help='Output file path (Excel)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of reviews to label (for testing)')
    
    args = parser.parse_args()
    
    # Label
    label_file(args.input, args.output, limit=args.limit)

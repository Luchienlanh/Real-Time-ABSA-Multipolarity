"""
Auto-labeling Script for ABSA Data
Uses Ollama/Mistral to label Vietnamese e-commerce reviews based on annotation guidelines.

Label Values:
- 1: Positive (tích cực)
- 0: Neutral (trung lập)  
- -1: Negative (tiêu cực)
- 2: Not mentioned (không nhắc đến)
- [-1,1]: Multi-polarity (vừa tích cực vừa tiêu cực)
"""

import os
import sys
import pandas as pd
import json
import time
import requests
from tqdm import tqdm
from typing import Dict, List, Optional
import re

# Configuration
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "mistral"  # or "llama2", "phi"
BATCH_SIZE = 1  # Process one at a time for accuracy

# 9 Aspects
ASPECTS = [
    'Chất lượng sản phẩm',       # Quality, durability, materials
    'Hiệu năng & Trải nghiệm',   # Performance, user experience  
    'Đúng mô tả',                # Accuracy of description
    'Giá cả & Khuyến mãi',       # Price, discounts, value
    'Vận chuyển',                # Shipping speed, delivery
    'Đóng gói',                  # Packaging quality
    'Dịch vụ & Thái độ Shop',    # Customer service, seller attitude
    'Bảo hành & Đổi trả',        # Warranty, returns
    'Tính xác thực',             # Authenticity (fake/genuine)
]

# Keywords for each aspect (for rule-based fallback)
ASPECT_KEYWORDS = {
    'Chất lượng sản phẩm': {
        'positive': ['đẹp', 'tốt', 'chất lượng', 'bền', 'chắc', 'mịn', 'đẹp lắm', 'ổn', 'ok', 'ưng'],
        'negative': ['xấu', 'tệ', 'kém', 'mỏng', 'dở', 'lỗi', 'hỏng', 'rách', 'bong', 'tróc']
    },
    'Hiệu năng & Trải nghiệm': {
        'positive': ['dùng tốt', 'xài được', 'sử dụng ok', 'chạy mượt', 'nhanh', 'êm'],
        'negative': ['dùng tệ', 'khó dùng', 'chậm', 'lag', 'nóng', 'hao pin', 'khó sử dụng']
    },
    'Đúng mô tả': {
        'positive': ['đúng hình', 'giống hình', 'như mô tả', 'đúng mẫu', 'đúng size', 'như ảnh'],
        'negative': ['khác hình', 'không giống', 'sai màu', 'sai size', 'không như', 'lừa đảo']
    },
    'Giá cả & Khuyến mãi': {
        'positive': ['rẻ', 'giá tốt', 'hợp lý', 'đáng tiền', 'hời', 'giá ok', 'sale'],
        'negative': ['đắt', 'mắc', 'giá cao', 'không đáng', 'chặt chém']
    },
    'Vận chuyển': {
        'positive': ['giao nhanh', 'ship nhanh', 'nhanh lắm', 'shipper tốt', 'đúng hẹn'],
        'negative': ['giao chậm', 'ship chậm', 'trễ', 'lâu', 'delay', 'đợi lâu']
    },
    'Đóng gói': {
        'positive': ['đóng gói cẩn thận', 'gói kỹ', 'đóng gói đẹp', 'bọc kỹ', 'an toàn'],
        'negative': ['đóng gói sơ sài', 'móp', 'bẹp', 'hư hộp', 'không cẩn thận']
    },
    'Dịch vụ & Thái độ Shop': {
        'positive': ['shop nhiệt tình', 'tư vấn tốt', 'shop ok', 'seller tốt', 'thân thiện'],
        'negative': ['shop tệ', 'thái độ kém', 'không nhiệt tình', 'không rep']
    },
    'Bảo hành & Đổi trả': {
        'positive': ['đổi nhanh', 'hoàn tiền', 'bảo hành tốt', 'hỗ trợ đổi'],
        'negative': ['không đổi', 'không bảo hành', 'từ chối đổi', 'không hoàn']
    },
    'Tính xác thực': {
        'positive': ['chính hãng', 'hàng thật', 'auth', 'real', 'xịn'],
        'negative': ['hàng giả', 'fake', 'nhái', 'không phải hàng thật', 'hàng dỏm']
    }
}


def create_prompt(review: str) -> str:
    """Create prompt for LLM to label the review."""
    return f"""Bạn là chuyên gia phân tích cảm xúc e-commerce. Hãy đánh nhãn bình luận sau theo 9 khía cạnh.

BÌNH LUẬN: "{review}"

9 KHÍA CẠNH:
1. Chất lượng sản phẩm - vật liệu, độ bền, form dáng
2. Hiệu năng & Trải nghiệm - khi sử dụng, đi, mặc
3. Đúng mô tả - so với hình ảnh/mô tả
4. Giá cả & Khuyến mãi - giá trị, rẻ/đắt
5. Vận chuyển - tốc độ giao hàng, shipper
6. Đóng gói - bao bì, đóng gói
7. Dịch vụ & Thái độ Shop - CSKH, seller
8. Bảo hành & Đổi trả - đổi trả, hoàn tiền
9. Tính xác thực - hàng thật/giả

GIÁ TRỊ NHÃN:
- 1: Tích cực (khen)
- 0: Trung lập (nhắc đến nhưng không rõ cảm xúc)
- -1: Tiêu cực (chê)
- 2: Không nhắc đến
- [-1,1]: Vừa khen vừa chê (ví dụ: "đẹp nhưng mỏng")

Trả về CHÍNH XÁC JSON như sau:
{{"Chất lượng sản phẩm": 2, "Hiệu năng & Trải nghiệm": 2, "Đúng mô tả": 2, "Giá cả & Khuyến mãi": 2, "Vận chuyển": 2, "Đóng gói": 2, "Dịch vụ & Thái độ Shop": 2, "Bảo hành & Đổi trả": 2, "Tính xác thực": 2}}

Nếu một khía cạnh vừa được khen VÀ chê, dùng "[-1,1]" (có dấu ngoặc vuông).
CHỈ TRẢ VỀ JSON, KHÔNG GIẢI THÍCH."""


def call_ollama(prompt: str, timeout: int = 60) -> Optional[str]:
    """Call Ollama API."""
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=timeout
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            print(f"Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Ollama error: {e}")
        return None


def parse_llm_response(response: str) -> Dict[str, any]:
    """Parse LLM response to extract labels."""
    try:
        # Try direct JSON parse
        data = json.loads(response)
        return data
    except:
        # Try to find JSON in response
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return None


def rule_based_label(review: str) -> Dict[str, any]:
    """Rule-based labeling as fallback."""
    review_lower = review.lower()
    result = {asp: 2 for asp in ASPECTS}  # Default: not mentioned
    
    for aspect, keywords in ASPECT_KEYWORDS.items():
        pos_found = any(kw in review_lower for kw in keywords['positive'])
        neg_found = any(kw in review_lower for kw in keywords['negative'])
        
        if pos_found and neg_found:
            result[aspect] = "[-1,1]"
        elif pos_found:
            result[aspect] = 1
        elif neg_found:
            result[aspect] = -1
    
    # Handle generic positive/negative
    if any(w in review_lower for w in ['tốt', 'đẹp', 'ok', 'ổn', 'ưng', 'thích']):
        if result['Chất lượng sản phẩm'] == 2:
            result['Chất lượng sản phẩm'] = 1
    
    if any(w in review_lower for w in ['tệ', 'xấu', 'dở', 'thất vọng']):
        if result['Chất lượng sản phẩm'] == 2:
            result['Chất lượng sản phẩm'] = -1
    
    return result


def label_review(review: str, use_llm: bool = True) -> Dict[str, any]:
    """Label a single review."""
    if not review or not review.strip():
        return {asp: 2 for asp in ASPECTS}
    
    if use_llm:
        prompt = create_prompt(review)
        response = call_ollama(prompt)
        
        if response:
            labels = parse_llm_response(response)
            if labels:
                # Validate and fill missing aspects
                for asp in ASPECTS:
                    if asp not in labels:
                        labels[asp] = 2
                return labels
    
    # Fallback to rule-based
    return rule_based_label(review)


def label_file(input_path: str, output_path: str, use_llm: bool = True, limit: int = None):
    """Label all reviews in a file."""
    print(f"\n Processing: {input_path}")
    
    df = pd.read_excel(input_path)
    total = len(df) if limit is None else min(limit, len(df))
    
    print(f"   Total reviews: {total}")
    
    # Initialize aspect columns
    for asp in ASPECTS:
        df[asp] = 2  # Default: not mentioned
    
    # Label each review
    for i in tqdm(range(total), desc="Labeling"):
        review = str(df.iloc[i]['reviewContent'])
        labels = label_review(review, use_llm=use_llm)
        
        for asp, val in labels.items():
            if asp in df.columns:
                df.at[i, asp] = val
        
        # Small delay to avoid rate limiting
        if use_llm and i % 10 == 0:
            time.sleep(0.5)
    
    # Save
    df.to_excel(output_path, index=False)
    print(f"    Saved to: {output_path}")
    return df


def label_all_test_flow(use_llm: bool = True, limit_per_file: int = None):
    """Label all test_flow files."""
    import glob
    
    folder = r'c:\SE363 (1)\data\test_flow'
    output_folder = r'c:\SE363 (1)\data\labeled'
    os.makedirs(output_folder, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(folder, 'test_flow_reviews_*.xlsx')))
    
    print(f" Starting auto-labeling for {len(files)} files")
    print(f"   Mode: {'LLM (Ollama)' if use_llm else 'Rule-based'}")
    print(f"   Output folder: {output_folder}")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, f"labeled_{filename}")
        
        try:
            label_file(file_path, output_path, use_llm=use_llm, limit=limit_per_file)
        except Exception as e:
            print(f"    Error: {e}")
    
    print("\n All files processed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-label ABSA data")
    parser.add_argument("--use-llm", action="store_true", default=False, help="Use LLM for labeling")
    parser.add_argument("--limit", type=int, default=None, help="Limit reviews per file")
    parser.add_argument("--file", type=str, default=None, help="Process single file")
    
    args = parser.parse_args()
    
    if args.file:
        output = args.file.replace('.xlsx', '_labeled.xlsx')
        label_file(args.file, output, use_llm=args.use_llm, limit=args.limit)
    else:
        label_all_test_flow(use_llm=args.use_llm, limit_per_file=args.limit)

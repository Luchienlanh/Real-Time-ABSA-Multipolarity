"""
PhoGPT-based ABSA Auto-Labeling Tool (Pure LLM Approach)
Uses VinAI PhoGPT for Vietnamese language understanding without keyword matching

Model: vinai/PhoGPT-7B5-Instruct (7.5B parameters)
Requirement: GPU with ~16GB VRAM

Usage:
    # Test with small batch
    python scripts/phogpt_labeler.py --input data/rawdata/all_reviews_combined.xlsx --output data/phogpt_labeled/test.xlsx --limit 50
    
    # Full labeling
    python scripts/phogpt_labeler.py --input data/rawdata/all_reviews_combined.xlsx --output data/phogpt_labeled/labeled_all.xlsx
"""

import os
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import re
from typing import Dict, List

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = "vinai/PhoGPT-7B5-Instruct"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1  # Low temperature for more deterministic output

# 9 Aspects
ASPECTS = [
    'Chất lượng sản phẩm',
    'Hiệu năng & Trải nghiệm',
    'Đúng mô tả',
    'Giá cả & Khuyến mãi',
    'Vận chuyển',
    'Đóng gói',
    'Dịch vụ & Thái độ Shop',
    'Bảo hành & Đổi trả',
    'Tính xác thực',
]


class PhoGPTLabeler:
    """PhoGPT-based ABSA labeler using generation and prompt engineering."""
    
    def __init__(self, use_8bit=False):
        """
        Initialize PhoGPT model.
        
        Args:
            use_8bit: If True, load model in 8-bit quantization (saves memory)
        """
        print(f"\n{'='*60}")
        print(f"Loading PhoGPT model: {MODEL_NAME}")
        print(f"Device: {DEVICE}")
        print(f"8-bit quantization: {use_8bit}")
        print(f"{'='*60}\n")
        
        if not torch.cuda.is_available():
            print("⚠️  WARNING: No GPU detected! PhoGPT will be VERY slow on CPU.")
            print("   Consider using Google Colab with GPU or smaller model.")
            response = input("   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True
        )
        
        # Load model with optimizations
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        
        if use_8bit and torch.cuda.is_available():
            load_kwargs["load_in_8bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            **load_kwargs
        )
        
        if not use_8bit and torch.cuda.is_available():
            self.model = self.model.to(DEVICE)
        
        self.model.eval()
        
        print("✅ Model loaded successfully!\n")
    
    def create_prompt(self, review: str) -> str:
        """
        Create prompt for PhoGPT based on annotation guideline.
        
        Args:
            review: Review text to analyze
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""### Instruction:
Bạn là chuyên gia phân tích cảm xúc cho bình luận thương mại điện tử.
Nhiệm vụ: Phân tích bình luận sau theo 9 khía cạnh và gán nhãn cảm xúc.

**9 Khía cạnh:**
1. Chất lượng sản phẩm - chất liệu, độ bền, form dáng
2. Hiệu năng & Trải nghiệm - trải nghiệm sử dụng, hiệu suất
3. Đúng mô tả - độ chính xác so với hình ảnh/mô tả
4. Giá cả & Khuyến mãi - giá trị, ưu đãi
5. Vận chuyển - tốc độ, chất lượng giao hàng
6. Đóng gói - bao bì, đóng gói sản phẩm
7. Dịch vụ & Thái độ Shop - CSKH, thái độ người bán
8. Bảo hành & Đổi trả - chính sách bảo hành, đổi trả
9. Tính xác thực - hàng thật/giả, nguồn gốc

**Nhãn cảm xúc:**
- 1: Tích cực (khen, hài lòng)
- 0: Trung lập (nhắc đến nhưng không rõ cảm xúc)
- -1: Tiêu cực (chê, không hài lòng)
- 2: Không nhắc đến
- [-1,1]: Đa cực (vừa khen vừa chê cùng khía cạnh)

**Bình luận:**
"{review}"

**Yêu cầu:**
- Đọc kỹ toàn bộ bình luận
- Xác định từng khía cạnh được đề cập
- Gán nhãn cảm xúc chính xác
- Chú ý trường hợp đa cực (có từ "nhưng", "mà")
- Trả về CHÍNH XÁC định dạng JSON bên dưới, KHÔNG giải thích thêm

### Response:
```json
{{
  "Chất lượng sản phẩm": <nhãn>,
  "Hiệu năng & Trải nghiệm": <nhãn>,
  "Đúng mô tả": <nhãn>,
  "Giá cả & Khuyến mãi": <nhãn>,
  "Vận chuyển": <nhãn>,
  "Đóng gói": <nhãn>,
  "Dịch vụ & Thái độ Shop": <nhãn>,
  "Bảo hành & Đổi trả": <nhãn>,
  "Tính xác thực": <nhãn>
}}
```"""
        
        return prompt
    
    def parse_response(self, response: str) -> Dict[str, any]:
        """
        Parse PhoGPT response to extract labels.
        
        Args:
            response: Raw response from model
            
        Returns:
            Dict mapping aspect to label
        """
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                labels = json.loads(json_str)
                
                # Validate and fill missing aspects
                result = {}
                for aspect in ASPECTS:
                    if aspect in labels:
                        result[aspect] = labels[aspect]
                    else:
                        result[aspect] = 2  # Default: not mentioned
                
                return result
        except Exception as e:
            print(f"   ⚠️  Parse error: {e}")
            print(f"   Response: {response[:200]}")
        
        # Fallback: all not mentioned
        return {asp: 2 for asp in ASPECTS}
    
    def label_review(self, review: str) -> Dict[str, any]:
        """
        Label a single review using PhoGPT.
        
        Args:
            review: Review text
            
        Returns:
            Dict mapping aspect to label
        """
        if not review or not review.strip():
            return {asp: 2 for asp in ASPECTS}
        
        # Create prompt
        prompt = self.create_prompt(review)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after prompt)
        response = response[len(prompt):]
        
        # Parse labels
        labels = self.parse_response(response)
        
        return labels
    
    def label_dataframe(self, df: pd.DataFrame, limit: int = None, batch_size: int = 1) -> pd.DataFrame:
        """
        Label entire dataframe.
        
        Args:
            df: DataFrame with 'reviewContent' column
            limit: If set, only label first N rows
            batch_size: Number of reviews to process before saving checkpoint (not actual batch inference)
            
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
        print(f"\nLabeling {len(df)} reviews with PhoGPT...")
        print(f"This may take a while (~10-30s per review depending on GPU)\n")
        
        for idx in tqdm(range(len(df)), desc="Labeling"):
            review = str(df.iloc[idx]['reviewContent'])
            
            try:
                labels = self.label_review(review)
                
                # Update dataframe
                for aspect, label in labels.items():
                    df.at[idx, aspect] = label
            
            except Exception as e:
                print(f"\n   Error at row {idx}: {e}")
                # Keep default labels (2)
                continue
        
        return df


def label_file(input_path: str, output_path: str, limit: int = None, use_8bit: bool = False):
    """Label a single file with PhoGPT."""
    print(f"\n{'='*60}")
    print(f"PhoGPT-based ABSA Auto-Labeling")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    if limit:
        print(f"Limit:  {limit} reviews (for testing)")
    print(f"{'='*60}\n")
    
    # Read input
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)
    
    print(f"Total reviews in file: {len(df)}")
    
    # Create labeler
    labeler = PhoGPTLabeler(use_8bit=use_8bit)
    
    # Label
    df_labeled = labeler.label_dataframe(df, limit=limit)
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df_labeled.to_excel(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Saved to: {output_path}")
    print(f"{'='*60}")
    
    # Statistics
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
    
    parser = argparse.ArgumentParser(description="PhoGPT-based ABSA Auto-Labeling (Pure LLM)")
    parser.add_argument('--input', type=str, required=True, help='Input file (Excel/CSV)')
    parser.add_argument('--output', type=str, required=True, help='Output file (Excel)')
    parser.add_argument('--limit', type=int, default=None, help='Limit reviews (for testing)')
    parser.add_argument('--use-8bit', action='store_true', help='Use 8-bit quantization (saves memory)')
    
    args = parser.parse_args()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: No GPU detected!")
        print("   PhoGPT-7B5 is a large model and will be VERY slow on CPU.")
        print("   Estimated time: ~5-10 minutes per review on CPU")
        print("\n   Recommendations:")
        print("   - Use Google Colab with GPU (free)")
        print("   - Use Gemini API instead (faster)")
        print("   - Rent GPU cloud instance")
        print("\n")
    
    # Label
    label_file(
        args.input, 
        args.output, 
        limit=args.limit,
        use_8bit=args.use_8bit
    )

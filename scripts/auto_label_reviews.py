"""
Auto-labeling script for e-commerce reviews using keyword-based ABSA.
Based on the annotation guideline for Real-Time Streaming Sentiment Analysis.
"""

import pandas as pd
import re
from typing import List, Tuple

# Define 9 aspects with Vietnamese keywords
ASPECTS = [
    'Chất lượng sản phẩm',
    'Hiệu năng & Trải nghiệm', 
    'Đúng mô tả',
    'Giá cả & Khuyến mãi',
    'Vận chuyển',
    'Đóng gói',
    'Dịch vụ & Thái độ Shop',
    'Bảo hành & Đổi trả',
    'Tính xác thực'
]

# Keywords mapping for aspect detection
ASPECT_KEYWORDS = {
    'Chất lượng sản phẩm': [
        'chất lượng', 'chất', 'vải', 'da', 'form', 'bền', 'cứng', 'mềm', 'mịn', 'thô', 
        'nặng', 'nhẹ', 'đẹp', 'xấu', 'tốt', 'tệ', 'hàng', 'sản phẩm', 'sp', 'dày', 'mỏng',
        'rách', 'hỏng', 'gãy', 'bung', 'lỗi', 'bị', 'còn', 'màu', 'size', 'kích thước',
        'chất liệu', 'material', 'quality', 'tóc', 'mượt', 'khô', 'rít', 'xơ', 'bết',
        'dầu gội', 'kem', 'mùi', 'hôi', 'thơm'
    ],
    'Hiệu năng & Trải nghiệm': [
        'dùng', 'xài', 'sử dụng', 'trải nghiệm', 'hoạt động', 'chạy', 'pin', 'nhanh', 
        'chậm', 'mượt', 'lag', 'giật', 'dễ', 'khó', 'tiện', 'tiện lợi', 'hiệu quả',
        'kết quả', 'công dụng', 'tác dụng', 'gội', 'rửa', 'dán', 'lắp', 'mang', 'đi',
        'chơi', 'đọc', 'xem', 'nghe', 'cảm giác', 'thoải mái', 'khó chịu', 'đau'
    ],
    'Đúng mô tả': [
        'giống hình', 'như hình', 'đúng mô tả', 'giống ảnh', 'như ảnh', 'đúng', 'khác',
        'không giống', 'khác hình', 'quảng cáo', 'mô tả', 'description', 'đúng mẫu',
        'như quảng cáo', 'giống quảng cáo', 'đúng loại', 'sai', 'nhầm', 'lộn'
    ],
    'Giá cả & Khuyến mãi': [
        'giá', 'tiền', 'rẻ', 'đắt', 'hời', 'voucher', 'mã giảm', 'sale', 'khuyến mãi',
        'ưu đãi', 'giảm giá', 'freeship', 'miễn phí', 'giá trị', 'đáng tiền', 'xứng đáng',
        'tầm giá', 'trong tầm', 'túi tiền', 'price', 'cheap', 'expensive', 'phí'
    ],
    'Vận chuyển': [
        'ship', 'giao', 'giao hàng', 'shipper', 'vận chuyển', 'đơn vị', 'nhanh', 'chậm',
        'trễ', 'đúng hẹn', 'lâu', 'nhận hàng', 'giao đúng', 'giao sai', 'chuyển hàng',
        'delivery', 'shipping', 'thời gian giao'
    ],
    'Đóng gói': [
        'đóng gói', 'hộp', 'bọc', 'cẩn thận', 'bubble', 'móp', 'bẹp', 'vỡ', 'bao bì',
        'gói hàng', 'đóng hộp', 'packaging', 'kĩ', 'kỹ', 'cẩu thả', 'sơ sài', 'seal'
    ],
    'Dịch vụ & Thái độ Shop': [
        'shop', 'seller', 'hỗ trợ', 'tư vấn', 'nhiệt tình', 'trả lời', 'phản hồi',
        'thái độ', 'cskh', 'chăm sóc', 'uy tín', 'tin cậy', 'người bán', 'nhân viên',
        'service', 'sao', 'điểm', 'ủng hộ', 'quay lại', 'lần sau'
    ],
    'Bảo hành & Đổi trả': [
        'bảo hành', 'đổi', 'trả', 'hoàn', 'lỗi', 'sửa chữa', 'warranty', 'return',
        'exchange', 'refund', 'đổi size', 'đổi màu', 'hoàn tiền'
    ],
    'Tính xác thực': [
        'chính hãng', 'auth', 'fake', 'nhái', 'real', 'thật', 'giả', 'authentic',
        'original', 'hàng thật', 'hàng giả', 'nguồn gốc', 'xuất xứ', 'uy tín'
    ]
}

# Positive sentiment keywords
POSITIVE_KEYWORDS = [
    'tốt', 'đẹp', 'ưng', 'thích', 'ok', 'ổn', 'nhanh', 'chất lượng', 'xuất sắc',
    'tuyệt vời', 'tuyệt', 'hài lòng', 'mượt', 'mịn', 'thơm', 'bền', 'cứng cáp',
    'chắc chắn', 'đáng mua', 'nên mua', 'recommend', 'rẻ', 'hời', 'xứng đáng',
    'cảm ơn', 'nhiệt tình', 'cẩn thận', 'kỹ', 'kĩ', 'vui', 'dễ', 'tiện', 'tiện lợi',
    'nice', 'good', 'great', 'amazing', 'love', 'perfect', 'excellent', 'yêu',
    'ủng hộ', '5 sao', '10 điểm', 'quay lại', 'mua lại', 'mua thêm', 'giống hình',
    'đúng', 'vừa', 'fit', 'phù hợp', 'như mong đợi', 'như kỳ vọng'
]

# Negative sentiment keywords  
NEGATIVE_KEYWORDS = [
    'tệ', 'xấu', 'dở', 'chậm', 'lỗi', 'hỏng', 'thất vọng', 'không hài lòng',
    'không thích', 'ghét', 'rách', 'vỡ', 'móp', 'bẹp', 'giả', 'fake', 'nhái',
    'không đúng', 'sai', 'nhầm', 'khác', 'không giống', 'mất', 'thiếu', 'chán',
    'không ok', 'không ổn', 'tồi', 'kém', 'bad', 'terrible', 'awful', 'hate',
    'worst', 'poor', 'đắt', 'mắc', 'không đáng', 'phí tiền', 'tiếc', 'hối hận',
    'không nên mua', 'không recommend', 'cẩu thả', 'sơ sài', 'thô', 'cứng',
    'đau', 'khó chịu', 'ngứa', 'rít', 'xơ', 'bết', 'khô', 'hôi', 'kinh khủng',
    'ghê', 'không mua', 'không lần sau', 'chê', 'phàn nàn', 'bung', 'gãy',
    'không trl', 'không trả lời', 'không hỗ trợ', 'lừa', 'lừa đảo'
]

# Neutral indicators
NEUTRAL_KEYWORDS = [
    'tạm', 'bình thường', 'cũng được', 'ok thôi', 'được', 'chưa biết', 'chưa dùng',
    'chờ xem', 'sẽ xem', 'tiền nào của nấy', 'so so', 'thường', 'trung bình'
]


def detect_aspect_sentiment(text: str) -> dict:
    """
    Detect aspects and their sentiments from review text.
    Returns dict with aspect -> label (1, 0, -1, 2, or list like [-1,1])
    """
    text_lower = text.lower()
    results = {}
    
    for aspect in ASPECTS:
        # Check if aspect is mentioned
        aspect_keywords = ASPECT_KEYWORDS.get(aspect, [])
        aspect_mentioned = any(kw in text_lower for kw in aspect_keywords)
        
        if not aspect_mentioned:
            results[aspect] = 2  # Not mentioned
            continue
        
        # Detect sentiment for this aspect
        pos_score = 0
        neg_score = 0
        neu_score = 0
        
        # Find relevant sentences/phrases for this aspect
        for kw in aspect_keywords:
            if kw in text_lower:
                # Check surrounding context for sentiment
                idx = text_lower.find(kw)
                context_start = max(0, idx - 50)
                context_end = min(len(text_lower), idx + len(kw) + 50)
                context = text_lower[context_start:context_end]
                
                # Count sentiment keywords in context
                for pos_kw in POSITIVE_KEYWORDS:
                    if pos_kw in context:
                        pos_score += 1
                        
                for neg_kw in NEGATIVE_KEYWORDS:
                    if neg_kw in context:
                        neg_score += 1
                        
                for neu_kw in NEUTRAL_KEYWORDS:
                    if neu_kw in context:
                        neu_score += 1
        
        # Additional: check for "nhưng", "tuy nhiên" patterns indicating mixed sentiment
        has_contrast = any(word in text_lower for word in ['nhưng', 'tuy nhiên', 'tuy', 'mặc dù', 'dù'])
        
        # Determine final sentiment
        if pos_score > 0 and neg_score > 0:
            # Multi-polarity
            results[aspect] = "[-1,1]"
        elif pos_score > 0 and neu_score > 0 and neg_score == 0:
            if has_contrast:
                results[aspect] = "[0,1]"
            else:
                results[aspect] = 1
        elif neg_score > 0 and neu_score > 0 and pos_score == 0:
            if has_contrast:
                results[aspect] = "[-1,0]"
            else:
                results[aspect] = -1
        elif pos_score > neg_score:
            results[aspect] = 1
        elif neg_score > pos_score:
            results[aspect] = -1
        elif neu_score > 0:
            results[aspect] = 0
        else:
            # Default: if mentioned but no clear sentiment, use star rating context
            results[aspect] = 0
    
    return results


def label_reviews(input_file: str, output_file: str):
    """Label all reviews in the input file and save to output."""
    
    # Read input
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file, encoding='utf-8')
    else:
        df = pd.read_excel(input_file)
    
    print(f"Loaded {len(df)} reviews from {input_file}")
    
    # Get the content column (try different names)
    content_col = None
    for col in ['content', 'reviewContent', 'review', 'text', 'comment']:
        if col in df.columns:
            content_col = col
            break
    
    if content_col is None:
        raise ValueError(f"No content column found. Available columns: {df.columns.tolist()}")
    
    print(f"Using content column: {content_col}")
    
    # Rename to standard format
    df['reviewContent'] = df[content_col].fillna('')
    
    # Initialize aspect columns
    for aspect in ASPECTS:
        df[aspect] = 2  # Default: not mentioned
    
    # Label each review
    for idx, row in df.iterrows():
        text = str(row['reviewContent'])
        
        # Use star rating as additional context
        star = row.get('star', 3)
        
        # Get aspect sentiments
        sentiments = detect_aspect_sentiment(text)
        
        # Apply sentiments
        for aspect, sentiment in sentiments.items():
            df.at[idx, aspect] = sentiment
        
        # If no aspects detected but review is short positive/negative
        all_not_mentioned = all(df.at[idx, asp] == 2 for asp in ASPECTS)
        if all_not_mentioned and len(text) < 100:
            # Check overall sentiment
            text_lower = text.lower()
            has_pos = any(kw in text_lower for kw in POSITIVE_KEYWORDS)
            has_neg = any(kw in text_lower for kw in NEGATIVE_KEYWORDS)
            
            if has_pos or has_neg:
                # Assign to "Chất lượng sản phẩm" as default
                if has_pos and has_neg:
                    df.at[idx, 'Chất lượng sản phẩm'] = "[-1,1]"
                elif has_pos:
                    df.at[idx, 'Chất lượng sản phẩm'] = 1
                elif has_neg:
                    df.at[idx, 'Chất lượng sản phẩm'] = -1
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} reviews...")
    
    # Select output columns
    output_cols = ['reviewContent'] + ASPECTS
    output_df = df[output_cols]
    
    # Save output
    if output_file.endswith('.csv'):
        output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    else:
        output_df.to_excel(output_file, index=False, engine='openpyxl')
    
    print(f"Saved labeled data to {output_file}")
    
    # Print statistics
    print("\n=== Labeling Statistics ===")
    for aspect in ASPECTS:
        counts = output_df[aspect].value_counts()
        print(f"\n{aspect}:")
        for val, count in counts.items():
            pct = count / len(output_df) * 100
            print(f"  {val}: {count} ({pct:.1f}%)")
    
    return output_df


if __name__ == "__main__":
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/lazada_reviews.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "data/lazada_reviews_labeled.xlsx"
    
    label_reviews(input_file, output_file)

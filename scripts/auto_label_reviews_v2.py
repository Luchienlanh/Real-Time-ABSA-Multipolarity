"""
IMPROVED Auto-labeling script for e-commerce reviews using careful keyword-based ABSA.
Version 2.0 - More careful and thorough labeling based on annotation guideline.

Key improvements:
1. More comprehensive keyword lists for Vietnamese e-commerce
2. Better context analysis for sentiment detection
3. More accurate multi-polarity detection using "nhưng" patterns
4. Semantic grouping of keywords by sub-aspects
"""

import pandas as pd
import re
import os
from typing import List, Tuple, Dict, Union

# Define 9 aspects 
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

# ============================================================
# COMPREHENSIVE KEYWORD MAPPINGS
# ============================================================

ASPECT_KEYWORDS = {
    'Chất lượng sản phẩm': {
        'nouns': [
            'chất lượng', 'chất liệu', 'vải', 'da', 'form', 'kiểu dáng', 'màu', 'màu sắc',
            'size', 'kích thước', 'kích cỡ', 'áo', 'quần', 'giày', 'dép', 'váy', 'đầm',
            'túi', 'ví', 'ba lô', 'mũ', 'nón', 'khăn', 'găng tay', 'tất', 'bít tất',
            'sản phẩm', 'sp', 'hàng', 'đồ', 'món', 'cái', 'bộ', 'set',
            'chỉ', 'đường may', 'khóa', 'nút', 'dây', 'quai', 'đế', 'gót',
            'mùi', 'hương', 'tóc', 'da mặt', 'kem', 'dầu gội', 'sữa tắm',
            'chất', 'material', 'quality', 'product'
        ],
        'positive': [
            'đẹp', 'tốt', 'chất', 'xịn', 'xịn sò', 'ổn', 'ok', 'okie', 'được', 'ưng',
            'thích', 'mịn', 'mềm', 'mượt', 'bền', 'chắc', 'cứng cáp', 'dày dặn',
            'đẹp lắm', 'rất đẹp', 'quá đẹp', 'xinh', 'xinh xắn', 'sang', 'sang trọng',
            'cao cấp', 'premium', 'xịn xò', 'chất lượng', 'hoàn hảo', 'tuyệt vời',
            'xuất sắc', 'perfect', 'great', 'good', 'nice', 'amazing', 'yêu', 'love',
            'thơm', 'dễ chịu', 'mát', 'thoáng', 'êm', 'nhẹ nhàng', 'dễ thương', 'cute'
        ],
        'negative': [
            'xấu', 'tệ', 'kém', 'dở', 'tồi', 'thất vọng', 'không đẹp', 'không tốt',
            'mỏng', 'rẻ tiền', 'giả', 'nhái', 'dỏm', 'lởm', 'cùi', 'bèo',
            'rách', 'hỏng', 'vỡ', 'gãy', 'bung', 'tuột', 'bong', 'tróc', 'phai',
            'thô', 'cứng', 'nhám', 'xù', 'xơ', 'bết', 'khô', 'rít',
            'hôi', 'khó chịu', 'ngứa', 'kích ứng', 'dị ứng', 'nóng', 'bí',
            'nhăn', 'nhàu', 'méo', 'lệch', 'cong', 'vênh',
            'lỗi', 'bị lỗi', 'có lỗi', 'sai', 'nhầm', 'thiếu',
            'bad', 'terrible', 'awful', 'ugly', 'poor', 'cheap'
        ]
    },
    
    'Hiệu năng & Trải nghiệm': {
        'nouns': [
            'dùng', 'xài', 'sử dụng', 'trải nghiệm', 'cảm giác', 'cảm nhận',
            'mang', 'mặc', 'đi', 'đeo', 'chơi', 'gội', 'rửa', 'bôi', 'thoa',
            'hoạt động', 'chạy', 'pin', 'hiệu quả', 'công dụng', 'tác dụng', 'kết quả',
            'dán', 'lắp đặt', 'cài đặt', 'setup', 'performance', 'experience'
        ],
        'positive': [
            'tốt', 'ổn', 'ok', 'được', 'dễ', 'tiện', 'tiện lợi', 'nhanh', 'mượt',
            'êm', 'thoải mái', 'dễ chịu', 'vừa', 'vừa vặn', 'fit', 'phù hợp',
            'hiệu quả', 'công dụng tốt', 'như mong đợi', 'hài lòng',
            'thích', 'ưng', 'yêu', 'love', 'perfect', 'amazing'
        ],
        'negative': [
            'tệ', 'kém', 'dở', 'khó', 'khó chịu', 'không tiện', 'chậm', 'lag', 'giật',
            'đau', 'đau chân', 'phồng', 'rộp', 'chật', 'rộng', 'dài', 'ngắn',
            'không vừa', 'không fit', 'không phù hợp', 'không hiệu quả',
            'không tác dụng', 'không công dụng', 'thất vọng', 'chán',
            'không thích', 'ghét', 'awful', 'terrible', 'uncomfortable'
        ]
    },
    
    'Đúng mô tả': {
        'nouns': [
            'hình', 'ảnh', 'mô tả', 'quảng cáo', 'description', 'mẫu', 'kiểu',
            'như hình', 'giống hình', 'như ảnh', 'giống ảnh', 'theo hình',
            'đúng mô tả', 'đúng mẫu', 'đúng kiểu', 'đúng loại'
        ],
        'positive': [
            'đúng', 'giống', 'như', 'y hệt', 'y chang', 'chuẩn', 'chính xác',
            'đúng hình', 'giống hình', 'như hình', 'đúng mô tả', 'như mô tả',
            'đúng mẫu', 'giống mẫu', 'đúng quảng cáo', 'như quảng cáo'
        ],
        'negative': [
            'khác', 'không giống', 'không đúng', 'sai', 'nhầm', 'lộn', 'lệch',
            'khác hình', 'không giống hình', 'khác mô tả', 'không đúng mô tả',
            'khác mẫu', 'không giống mẫu', 'khác quảng cáo',
            'giao sai', 'gửi sai', 'đặt khác giao khác'
        ]
    },
    
    'Giá cả & Khuyến mãi': {
        'nouns': [
            'giá', 'tiền', 'price', 'cost', 'giá cả', 'giá tiền', 'giá trị',
            'voucher', 'mã giảm', 'coupon', 'sale', 'khuyến mãi', 'ưu đãi',
            'freeship', 'miễn phí ship', 'giảm giá', 'discount', 'deal'
        ],
        'positive': [
            'rẻ', 'hời', 'xứng đáng', 'đáng tiền', 'đáng giá', 'hợp lý', 'phải chăng',
            'tốt', 'ổn', 'được', 'ok', 'rẻ mà chất', 'rẻ mà đẹp',
            'trong tầm', 'vừa túi', 'tiết kiệm', 'worth', 'value',
            'giá tốt', 'giá ổn', 'giá hợp lý', 'tầm giá này'
        ],
        'negative': [
            'đắt', 'mắc', 'chát', 'không đáng', 'không xứng', 'phí tiền',
            'lừa đảo', 'lừa', 'chặt', 'chặt chém', 'đắt quá', 'mắc quá',
            'expensive', 'overpriced', 'rip off'
        ]
    },
    
    'Vận chuyển': {
        'nouns': [
            'ship', 'giao', 'giao hàng', 'vận chuyển', 'shipper', 'đơn vị vận chuyển',
            'chuyển hàng', 'gửi hàng', 'nhận hàng', 'delivery', 'shipping',
            'thời gian giao', 'ngày giao'
        ],
        'positive': [
            'nhanh', 'sớm', 'đúng hẹn', 'đúng ngày', 'siêu nhanh', 'cực nhanh',
            'tốt', 'ổn', 'ok', 'được', 'hài lòng', 'nhanh chóng',
            'giao nhanh', 'ship nhanh', 'giao sớm', 'nhận sớm'
        ],
        'negative': [
            'chậm', 'lâu', 'trễ', 'muộn', 'delay', 'quá chậm', 'quá lâu',
            'giao chậm', 'ship chậm', 'giao muộn', 'giao trễ',
            'mất hàng', 'thất lạc', 'không nhận được'
        ]
    },
    
    'Đóng gói': {
        'nouns': [
            'đóng gói', 'gói', 'hộp', 'bọc', 'bao bì', 'packaging', 'pack',
            'bubble', 'xốp', 'giấy', 'túi', 'thùng', 'carton'
        ],
        'positive': [
            'cẩn thận', 'kỹ', 'kĩ', 'chắc chắn', 'đẹp', 'gọn gàng', 'nguyên vẹn',
            'tốt', 'ổn', 'ok', 'được', 'cẩn thận', 'kỹ lưỡng', 'kĩ lưỡng',
            'đóng gói cẩn thận', 'gói kỹ', 'đóng hộp đẹp'
        ],
        'negative': [
            'cẩu thả', 'sơ sài', 'qua loa', 'không kỹ', 'không cẩn thận',
            'móp', 'bẹp', 'vỡ', 'rách', 'nát', 'hỏng', 'bị dập',
            'không bọc', 'không bubble', 'đóng gói tệ'
        ]
    },
    
    'Dịch vụ & Thái độ Shop': {
        'nouns': [
            'shop', 'seller', 'người bán', 'chủ shop', 'nhân viên', 'cskh',
            'dịch vụ', 'service', 'hỗ trợ', 'tư vấn', 'thái độ', 'phản hồi',
            'trả lời', 'trl', 'reply', 'response'
        ],
        'positive': [
            'tốt', 'ổn', 'ok', 'được', 'nhiệt tình', 'chu đáo', 'vui vẻ', 'thân thiện',
            'uy tín', 'tin cậy', 'reliable', 'nhanh nhẹn', 'chuyên nghiệp',
            'hỗ trợ tốt', 'tư vấn tốt', 'phản hồi nhanh', 'trả lời nhanh',
            '5 sao', '10 điểm', 'ủng hộ', 'quay lại', 'mua lại', 'lần sau'
        ],
        'negative': [
            'tệ', 'kém', 'dở', 'xấu', 'khó chịu', 'hách dịch', 'lạnh lùng',
            'không hỗ trợ', 'không tư vấn', 'không trả lời', 'không phản hồi',
            'lừa đảo', 'lừa', 'gian lận', 'thiếu trách nhiệm',
            'không uy tín', 'không tin cậy', 'không lần sau', 'không quay lại'
        ]
    },
    
    'Bảo hành & Đổi trả': {
        'nouns': [
            'bảo hành', 'đổi', 'trả', 'hoàn', 'return', 'exchange', 'refund',
            'warranty', 'đổi size', 'đổi màu', 'đổi sản phẩm', 'trả hàng',
            'hoàn tiền', 'sửa chữa', 'khiếu nại'
        ],
        'positive': [
            'được đổi', 'cho đổi', 'đổi được', 'hoàn tiền', 'bảo hành tốt',
            'đổi nhanh', 'đổi dễ', 'hỗ trợ đổi', 'cho trả'
        ],
        'negative': [
            'không đổi', 'không cho đổi', 'không trả', 'không hoàn',
            'không bảo hành', 'từ chối đổi', 'từ chối trả', 'khó đổi'
        ]
    },
    
    'Tính xác thực': {
        'nouns': [
            'chính hãng', 'auth', 'authentic', 'original', 'real', 'genuine',
            'fake', 'giả', 'nhái', 'hàng thật', 'hàng giả', 'hàng nhái',
            'nguồn gốc', 'xuất xứ', 'origin'
        ],
        'positive': [
            'chính hãng', 'auth', 'thật', 'real', 'authentic', 'original', 'genuine',
            'xịn', 'chuẩn', 'uy tín', 'đúng hãng'
        ],
        'negative': [
            'giả', 'fake', 'nhái', 'dỏm', 'lởm', 'không chính hãng',
            'hàng giả', 'hàng nhái', 'hàng dỏm', 'không auth'
        ]
    }
}

# Contrast words indicating potential multi-polarity
CONTRAST_WORDS = ['nhưng', 'tuy', 'tuy nhiên', 'mặc dù', 'dù', 'song', 'còn', 'nhưng mà', 'tuy vậy']


def normalize_text(text: str) -> str:
    """Normalize Vietnamese text for better matching."""
    text = str(text).lower().strip()
    # Common typo fixes
    replacements = {
        'ko ': 'không ', 'k ': 'không ', 'hok ': 'không ',
        'dc ': 'được ', 'đc ': 'được ', 'dk ': 'được ',
        'vs ': 'với ', 'j ': 'gì ',
        'sp ': 'sản phẩm ', 'trl ': 'trả lời ',
        ' r ': ' rồi ', ' ạ ': ' ', ' a ': ' ',
        'lun': 'luôn', 'nka': 'nha', 'nhé': 'nha',
        '. ': ' . ', ', ': ' , '
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def find_aspect_mentions(text: str, aspect: str) -> List[Tuple[int, str]]:
    """Find all mentions of an aspect in text with their positions."""
    mentions = []
    keywords = ASPECT_KEYWORDS[aspect]['nouns']
    text_lower = text.lower()
    
    for kw in keywords:
        start = 0
        while True:
            pos = text_lower.find(kw, start)
            if pos == -1:
                break
            mentions.append((pos, kw))
            start = pos + 1
    
    return mentions


def get_context_sentiment(text: str, position: int, aspect: str, window: int = 60) -> str:
    """
    Analyze sentiment in the context around a keyword position.
    Returns: 'positive', 'negative', 'neutral', or 'mixed'
    """
    text_lower = text.lower()
    start = max(0, position - window)
    end = min(len(text), position + window)
    context = text_lower[start:end]
    
    pos_keywords = ASPECT_KEYWORDS[aspect].get('positive', [])
    neg_keywords = ASPECT_KEYWORDS[aspect].get('negative', [])
    
    pos_count = sum(1 for kw in pos_keywords if kw in context)
    neg_count = sum(1 for kw in neg_keywords if kw in context)
    
    # Check for contrast words in context
    has_contrast = any(cw in context for cw in CONTRAST_WORDS)
    
    if pos_count > 0 and neg_count > 0:
        return 'mixed'
    elif has_contrast and (pos_count > 0 or neg_count > 0):
        return 'mixed'
    elif pos_count > neg_count:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    else:
        return 'neutral'


def analyze_aspect(text: str, aspect: str) -> Union[int, str]:
    """
    Carefully analyze an aspect in the text.
    Returns: 1 (positive), 0 (neutral), -1 (negative), 2 (not mentioned), 
             or multi-polarity like "[-1,1]"
    """
    text_normalized = normalize_text(text)
    mentions = find_aspect_mentions(text_normalized, aspect)
    
    if not mentions:
        # Check if any positive/negative keywords without explicit nouns
        pos_keywords = ASPECT_KEYWORDS[aspect].get('positive', [])
        neg_keywords = ASPECT_KEYWORDS[aspect].get('negative', [])
        
        has_pos = any(kw in text_normalized for kw in pos_keywords)
        has_neg = any(kw in text_normalized for kw in neg_keywords)
        
        if not has_pos and not has_neg:
            return 2  # Not mentioned
        
        # Use sentiment without explicit noun mention
        mentions = [(0, '')]
    
    # Collect sentiments from all mentions
    sentiments = set()
    
    for pos, kw in mentions:
        sentiment = get_context_sentiment(text_normalized, pos, aspect)
        sentiments.add(sentiment)
    
    # Also check global sentiment for this aspect
    pos_keywords = ASPECT_KEYWORDS[aspect].get('positive', [])
    neg_keywords = ASPECT_KEYWORDS[aspect].get('negative', [])
    
    global_pos = sum(1 for kw in pos_keywords if kw in text_normalized)
    global_neg = sum(1 for kw in neg_keywords if kw in text_normalized)
    
    # Check for explicit contrast patterns
    has_contrast = any(cw in text_normalized for cw in CONTRAST_WORDS)
    
    # Determine final label
    has_positive = 'positive' in sentiments or global_pos > 0
    has_negative = 'negative' in sentiments or global_neg > 0
    has_mixed = 'mixed' in sentiments or (has_contrast and has_positive and has_negative)
    
    if has_mixed or (has_positive and has_negative):
        return "[-1,1]"
    elif has_positive:
        return 1
    elif has_negative:
        return -1
    elif 'neutral' in sentiments or mentions:
        return 0
    else:
        return 2


def label_single_review(text: str) -> Dict[str, Union[int, str]]:
    """Label a single review with all 9 aspects."""
    results = {}
    text_normalized = normalize_text(text)
    
    for aspect in ASPECTS:
        results[aspect] = analyze_aspect(text_normalized, aspect)
    
    # Post-processing: handle very short reviews
    if len(text.strip()) < 15:
        # Very short review, likely overall comment
        text_lower = text.lower()
        
        # Generic positive
        if any(w in text_lower for w in ['tốt', 'đẹp', 'ổn', 'ok', 'được', 'ưng', 'thích', 'good', 'nice']):
            if all(v == 2 for v in results.values()):
                results['Chất lượng sản phẩm'] = 1
        
        # Generic negative
        elif any(w in text_lower for w in ['tệ', 'xấu', 'dở', 'kém', 'bad', 'terrible']):
            if all(v == 2 for v in results.values()):
                results['Chất lượng sản phẩm'] = -1
    
    return results


def label_all_reviews(input_dir: str, output_dir: str):
    """Label all reviews in CSV files from input directory."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files
    csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    total_reviews = 0
    
    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        output_path = os.path.join(output_dir, csv_file.replace('.csv', '_labeled.xlsx'))
        
        print(f"\nProcessing {csv_file}...")
        
        # Read input
        df = pd.read_csv(input_path, encoding='utf-8')
        
        # Find content column
        content_col = None
        for col in ['reviewContent', 'content', 'review', 'text', 'comment']:
            if col in df.columns:
                content_col = col
                break
        
        if content_col is None:
            content_col = df.columns[0]  # Use first column
        
        # Rename to standard
        df['reviewContent'] = df[content_col].fillna('')
        
        # Initialize aspect columns with object dtype to support mixed types
        for aspect in ASPECTS:
            df[aspect] = None
        
        # Label each review
        for idx, row in df.iterrows():
            text = str(row['reviewContent'])
            labels = label_single_review(text)
            
            for aspect, label in labels.items():
                df.at[idx, aspect] = label
            
            if (idx + 1) % 200 == 0:
                print(f"  Processed {idx + 1}/{len(df)}")
        
        # Select output columns
        output_cols = ['reviewContent'] + ASPECTS
        output_df = df[output_cols]
        
        # Save to Excel
        output_df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"  Saved {len(output_df)} reviews to {output_path}")
        
        total_reviews += len(output_df)
    
    print(f"\n{'='*50}")
    print(f"COMPLETED! Total: {total_reviews} reviews labeled")
    print(f"Output directory: {output_dir}")


def print_statistics(output_dir: str):
    """Print labeling statistics for all output files."""
    
    xlsx_files = [f for f in os.listdir(output_dir) if f.endswith('.xlsx')]
    
    # Aggregate counts
    aspect_counts = {asp: {'pos': 0, 'neg': 0, 'neu': 0, 'multi': 0, 'na': 0} for asp in ASPECTS}
    total = 0
    
    for xlsx_file in xlsx_files:
        df = pd.read_excel(os.path.join(output_dir, xlsx_file))
        total += len(df)
        
        for asp in ASPECTS:
            for val in df[asp]:
                val_str = str(val)
                if val_str == '1':
                    aspect_counts[asp]['pos'] += 1
                elif val_str == '-1':
                    aspect_counts[asp]['neg'] += 1
                elif val_str == '0':
                    aspect_counts[asp]['neu'] += 1
                elif '[' in val_str:
                    aspect_counts[asp]['multi'] += 1
                else:
                    aspect_counts[asp]['na'] += 1
    
    print(f"\n{'='*80}")
    print(f"LABELING STATISTICS - Total: {total} reviews")
    print(f"{'='*80}")
    print(f"{'Aspect':<30} {'Pos':>8} {'Neg':>8} {'Neu':>8} {'Multi':>8} {'N/A':>8} {'Mentioned':>10}")
    print("-" * 80)
    
    for asp in ASPECTS:
        c = aspect_counts[asp]
        mentioned = c['pos'] + c['neg'] + c['neu'] + c['multi']
        pct = mentioned / total * 100 if total > 0 else 0
        print(f"{asp:<30} {c['pos']:>8} {c['neg']:>8} {c['neu']:>8} {c['multi']:>8} {c['na']:>8} {pct:>9.1f}%")


if __name__ == "__main__":
    import sys
    
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "data/simple_process/split"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/simple_process/labeled"
    
    label_all_reviews(input_dir, output_dir)
    print_statistics(output_dir)

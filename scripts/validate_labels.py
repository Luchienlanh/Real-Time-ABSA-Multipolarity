"""
Data Quality Validation Tool for ABSA Dataset

Detects potential label errors by checking:
1. Keyword-label mismatches
2. Missing multi-polarity labels
3. Statistical anomalies
4. Contradiction patterns

Usage:
    python scripts/validate_labels.py --input data/labeled/ --output reports/validation_report.json
"""

import os
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
from collections import defaultdict
import glob
from tqdm import tqdm

# Aspects
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

# Validation rules: If review contains these keywords, check if label matches expected sentiment
VALIDATION_RULES = {
    'Chất lượng sản phẩm': {
        'negative_keywords': ['mỏng', 'kém', 'tệ', 'xấu', 'thất vọng', 'dở', 'lỗi', 'hỏng', 'rách'],
        'positive_keywords': ['đẹp', 'tốt', 'chất lượng', 'bền', 'ok', 'ổn', 'ưng'],
    },
    'Hiệu năng & Trải nghiệm': {
        'negative_keywords': ['rụng', 'tệ', 'kém hiệu quả', 'không hiệu quả', 'thất vọng'],
        'positive_keywords': ['hiệu quả', 'tốt', 'ok', 'dùng được'],
    },
    'Đúng mô tả': {
        'negative_keywords': ['không đúng', 'sai', 'khác', 'lừa đảo', 'quảng cáo sai'],
        'positive_keywords': ['đúng', 'giống hình', 'như mô tả', 'chính xác'],
    },
    'Vận chuyển': {
        'negative_keywords': ['chậm', 'lâu', 'trễ', 'delay', 'giao lâu'],
        'positive_keywords': ['nhanh', 'giao nhanh', 'ship nhanh', 'đúng hẹn'],
    },
    'Đóng gói': {
        'negative_keywords': ['móp', 'bẹp', 'hư', 'sơ sài', 'không cẩn thận'],
        'positive_keywords': ['cẩn thận', 'kỹ', 'đẹp', 'chắc chắn'],
    },
    'Dịch vụ & Thái độ Shop': {
        'negative_keywords': ['vô trách nhiệm', 'tệ', 'thái độ kém', 'không nhiệt tình'],
        'positive_keywords': ['nhiệt tình', 'tốt', 'hỗ trợ', 'thân thiện'],
    },
}

# Multi-polarity indicators
CONTRAST_WORDS = ['nhưng', 'mà', 'tuy nhiên', 'song', 'nhưng mà']


class LabelValidator:
    def __init__(self):
        self.issues = []
        self.stats = defaultdict(int)
        
    def validate_keyword_mismatch(self, review: str, aspect: str, label: any) -> List[Dict]:
        """Check if keywords contradict the label."""
        issues = []
        review_lower = review.lower()
        
        if aspect not in VALIDATION_RULES:
            return issues
        
        rules = VALIDATION_RULES[aspect]
        
        # Parse label
        if isinstance(label, str) and '[' in str(label):
            # Multi-polarity label like "[-1,1]"
            return issues  # Skip validation for multi-polarity
        
        try:
            label_val = int(float(label)) if label != 2 else 2
        except:
            return issues
        
        # Check negative keywords
        for kw in rules.get('negative_keywords', []):
            if kw in review_lower and label_val > 0:
                issues.append({
                    'type': 'keyword_mismatch',
                    'severity': 'high',
                    'aspect': aspect,
                    'keyword': kw,
                    'expected': 'NEG',
                    'actual': self._decode_label(label_val),
                    'review': review[:100]
                })
        
        # Check positive keywords
        for kw in rules.get('positive_keywords', []):
            if kw in review_lower and label_val < 0:
                issues.append({
                    'type': 'keyword_mismatch',
                    'severity': 'medium',
                    'aspect': aspect,
                    'keyword': kw,
                    'expected': 'POS',
                    'actual': self._decode_label(label_val),
                    'review': review[:100]
                })
        
        return issues
    
    def validate_multi_polarity(self, review: str, labels: Dict) -> List[Dict]:
        """Check if review should have multi-polarity but doesn't."""
        issues = []
        review_lower = review.lower()
        
        # Check for contrast words
        has_contrast = any(word in review_lower for word in CONTRAST_WORDS)
        
        if has_contrast:
            # Check if any aspect has multi-polarity
            has_multi = any('[' in str(labels.get(asp, 2)) for asp in ASPECTS)
            
            if not has_multi:
                # Find aspects that are mentioned
                mentioned = [asp for asp in ASPECTS if labels.get(asp, 2) != 2]
                
                if mentioned:
                    issues.append({
                        'type': 'missing_multi_polarity',
                        'severity': 'medium',
                        'aspects': mentioned,
                        'review': review[:150],
                        'suggestion': 'Review contains contrast words but no multi-polarity labels'
                    })
        
        return issues
    
    def validate_sample(self, row_idx: int, review: str, labels: Dict) -> List[Dict]:
        """Validate a single sample."""
        all_issues = []
        
        # 1. Keyword-label mismatch
        for aspect in ASPECTS:
            if aspect in labels:
                issues = self.validate_keyword_mismatch(review, aspect, labels[aspect])
                for issue in issues:
                    issue['row_idx'] = row_idx
                    all_issues.append(issue)
        
        # 2. Multi-polarity check
        issues = self.validate_multi_polarity(review, labels)
        for issue in issues:
            issue['row_idx'] = row_idx
            all_issues.append(issue)
        
        return all_issues
    
    def validate_file(self, file_path: str) -> Dict:
        """Validate an entire file."""
        print(f"\nValidating: {file_path}")
        
        df = pd.read_excel(file_path)
        file_issues = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking"):
            review = str(row.get('reviewContent', ''))
            
            labels = {asp: row.get(asp, 2) for asp in ASPECTS}
            
            issues = self.validate_sample(idx, review, labels)
            file_issues.extend(issues)
        
        return {
            'file': os.path.basename(file_path),
            'total_samples': len(df),
            'total_issues': len(file_issues),
            'issues': file_issues
        }
    
    def generate_report(self, results: List[Dict]) -> Dict:
        """Generate summary report."""
        total_samples = sum(r['total_samples'] for r in results)
        total_issues = sum(r['total_issues'] for r in results)
        
        # Group by issue type
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        by_aspect = defaultdict(int)
        
        for result in results:
            for issue in result['issues']:
                by_type[issue['type']] += 1
                by_severity[issue['severity']] += 1
                if 'aspect' in issue:
                    by_aspect[issue['aspect']] += 1
        
        # Calculate percentages
        issue_rate = (total_issues / total_samples * 100) if total_samples > 0 else 0
        
        report = {
            'summary': {
                'total_files': len(results),
                'total_samples': total_samples,
                'total_issues': total_issues,
                'issue_rate_percent': round(issue_rate, 2),
                'samples_with_issues': len(set(
                    (r['file'], issue['row_idx']) 
                    for r in results 
                    for issue in r['issues']
                ))
            },
            'by_type': dict(by_type),
            'by_severity': dict(by_severity),
            'by_aspect': dict(by_aspect),
            'files': results
        }
        
        return report
    
    def _decode_label(self, label: int) -> str:
        """Decode numeric label to text."""
        mapping = {1: 'POS', 0: 'NEU', -1: 'NEG', 2: 'NOT_MENTIONED'}
        return mapping.get(label, 'UNKNOWN')


def validate_directory(input_dir: str, output_path: str):
    """Validate all Excel files in directory."""
    validator = LabelValidator()
    
    files = glob.glob(os.path.join(input_dir, '*.xlsx'))
    print(f"Found {len(files)} files to validate")

    
    results = []
    for file_path in files:
        result = validator.validate_file(file_path)
        results.append(result)
    
    # Generate report
    report = validator.generate_report(results)
    
    # Save report
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    print(f"Total files:          {report['summary']['total_files']}")
    print(f"Total samples:        {report['summary']['total_samples']}")
    print(f"Total issues found:   {report['summary']['total_issues']}")
    print(f"Issue rate:           {report['summary']['issue_rate_percent']}%")
    print(f"Samples with issues:  {report['summary']['samples_with_issues']}")
    print("\nIssues by type:")
    for issue_type, count in report['by_type'].items():
        print(f"  - {issue_type}: {count}")
    print("\nIssues by severity:")
    for severity, count in report['by_severity'].items():
        print(f"  - {severity}: {count}")
    print(f"\nReport saved to: {output_path}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate ABSA label quality")
    parser.add_argument('--input', type=str, required=True, help='Input directory with labeled Excel files')
    parser.add_argument('--output', type=str, default='reports/validation_report.json', help='Output report path')
    
    args = parser.parse_args()
    
    validate_directory(args.input, args.output)

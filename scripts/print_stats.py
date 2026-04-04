import pandas as pd
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

output_dir = 'data/simple_process/labeled'
xlsx_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.xlsx')])

aspects = [
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

aspect_counts = {asp: {'pos': 0, 'neg': 0, 'neu': 0, 'multi': 0, 'na': 0} for asp in aspects}
total = 0

for xlsx_file in xlsx_files:
    df = pd.read_excel(os.path.join(output_dir, xlsx_file))
    total += len(df)
    
    for asp in aspects:
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

print(f"Total: {total} reviews")
print()
print("Aspect Statistics:")
print("-" * 70)

for i, asp in enumerate(aspects, 1):
    c = aspect_counts[asp]
    mentioned = c['pos'] + c['neg'] + c['neu'] + c['multi']
    pct = mentioned / total * 100 if total > 0 else 0
    print(f"{i}. Aspect: Pos={c['pos']}, Neg={c['neg']}, Neu={c['neu']}, Multi={c['multi']}, NA={c['na']} ({pct:.1f}%)")

# Show sample
print()
print("Sample labeled reviews:")
df1 = pd.read_excel(os.path.join(output_dir, xlsx_files[0]))
for i in range(3):
    review = str(df1['reviewContent'].iloc[i])[:100]
    q = df1['Chất lượng sản phẩm'].iloc[i]
    v = df1['Vận chuyển'].iloc[i]
    m = df1['Đúng mô tả'].iloc[i]
    print(f"  [{i+1}] {review}...")
    print(f"      Quality={q}, Shipping={v}, Description={m}")

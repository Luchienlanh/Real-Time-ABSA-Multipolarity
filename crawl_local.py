"""
LAZADA REVIEW CRAWLER - LOCAL VERSION (BALANCED RATINGS)
Chay tren Windows, khong can Google Colab
Cao review CAN BANG theo rating (1-5 sao)

Su dung: python crawl_local.py
"""
import os
import sys
import re
import json
import time
import random
import requests
import pandas as pd
import http.cookiejar as cookielib
from datetime import datetime
from fake_useragent import UserAgent

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8')

# ==================== CAU HINH ====================
COOKIE_FILE = "cookie/www.lazada.vn_cookies.txt"  # Duong dan file cookie
URL_FILE = "url/url_uncrawl.txt"                   # File chua danh sach URL
OUTPUT_DIR = "rawdata"                              # Thu muc luu ket qua
DELAY_MIN = 1.5                                     # Delay toi thieu (giay)
DELAY_MAX = 3                                       # Delay toi da (giay)

# So review mong muon cho MOI MUC RATING
# Script se cao toi da so nay cho moi rating (1-5 sao)
REVIEWS_PER_RATING = {
    1: 30,   # 1 sao - thuong it nen uu tien
    2: 30,   # 2 sao
    3: 30,   # 3 sao
    4: 30,   # 4 sao
    5: 30,   # 5 sao - thuong nhieu nhat
}
# ==================================================


def load_cookies(cookie_file):
    """Load cookies tu file Netscape format"""
    try:
        cj = cookielib.MozillaCookieJar()
        cj.load(cookie_file, ignore_discard=True, ignore_expires=True)
        print(f"[OK] Da load {len(cj)} cookies tu {cookie_file}")
        return cj
    except Exception as e:
        print(f"[X] Khong the load cookie: {e}")
        return None


def create_session(cookie_jar):
    """Tao session voi headers giong browser"""
    ua = UserAgent()
    session = requests.Session()
    session.cookies = cookie_jar
    session.headers.update({
        "User-Agent": ua.random,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.lazada.vn/",
        "Origin": "https://www.lazada.vn",
        "sec-ch-ua": '"Google Chrome";v="129", "Not=A?Brand";v="8"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
    })
    return session


def extract_item_id(url):
    """Lay item_id tu URL Lazada"""
    match = re.search(r'-i(\d+)', url)
    return match.group(1) if match else None


def crawl_reviews_by_rating(session, item_id, rating, max_reviews, delay_min, delay_max):
    """
    Cao review theo 1 muc rating cu the
    rating: 1-5 (filter value trong API)
    """
    reviews = []
    page = 1
    
    while len(reviews) < max_reviews:
        url = "https://my.lazada.vn/pdp/review/getReviewList"
        params = {
            "itemId": item_id,
            "pageSize": 50,
            "page": page,
            "filter": str(rating),  # Filter theo rating: 1, 2, 3, 4, 5
            "sort": "0"
        }

        try:
            r = session.get(url, params=params, timeout=30)

            if r.status_code != 200:
                break

            data = r.json()
            items = data.get("model", {}).get("items", [])

            if not items:
                break

            # Chi lay so luong can thiet
            needed = max_reviews - len(reviews)
            reviews.extend(items[:needed])
            
            total_available = data.get("model", {}).get("paging", {}).get("totalResults", 0)
            
            page += 1
            time.sleep(random.uniform(delay_min, delay_max))
            
            # Neu da het data hoac du so luong
            if len(items) < 50 or len(reviews) >= max_reviews:
                break

        except Exception as e:
            break
    
    return reviews


def crawl_lazada_reviews_balanced(session, product_url, reviews_per_rating, delay_min=1.5, delay_max=3):
    """
    Cao review CAN BANG theo tung muc rating (1-5 sao)
    Khong bi trung lap review
    """
    # Clean URL
    product_url = product_url.strip().strip('"').strip("'")
    
    item_id = extract_item_id(product_url)
    if not item_id:
        print(f"[X] Khong tim thay item_id trong URL: {product_url}")
        return None, None

    print(f"\n{'='*60}")
    print(f"[>] Bat dau cao item_id: {item_id}")
    print(f"    URL: {product_url}")
    print(f"    Mode: BALANCED RATINGS (1-5 sao)")
    print(f"{'='*60}")
    
    all_reviews = []
    seen_ids = set()  # Track review IDs da lay de khong bi trung
    rating_stats = {}
    
    # Cao theo tung rating tu 1 -> 5 sao
    for rating in [1, 2, 3, 4, 5]:
        target = reviews_per_rating.get(rating, 50)
        print(f"\n    [{rating} SAO] Dang cao (max: {target})...", end=" ")
        
        reviews = crawl_reviews_by_rating(
            session, item_id, rating, target, delay_min, delay_max
        )
        
        # Loc bo cac review da co (dua tren reviewRateId)
        unique_reviews = []
        for r in reviews:
            review_id = r.get('reviewRateId') or r.get('id') or hash(str(r.get('reviewContent', '')))
            if review_id not in seen_ids:
                seen_ids.add(review_id)
                unique_reviews.append(r)
        
        count = len(unique_reviews)
        rating_stats[rating] = count
        all_reviews.extend(unique_reviews)
        
        if count > 0:
            print(f"-> Lay duoc {count} reviews (unique)")
        else:
            print(f"-> Khong co review moi")
        
        # Delay giua cac rating
        time.sleep(random.uniform(0.5, 1))
    
    # Thong ke
    print(f"\n    {'='*40}")
    print(f"    THONG KE RATING (khong trung lap):")
    for r in [1, 2, 3, 4, 5]:
        count = rating_stats.get(r, 0)
        bar = '*' * min(count, 30)
        print(f"    {r} sao: {count:3d} | {bar}")
    print(f"    {'='*40}")
    print(f"    TONG: {len(all_reviews)} reviews (unique)")

    return all_reviews, item_id


def save_reviews(reviews, item_id, output_dir):
    """Luu reviews ra file Excel va JSON"""
    if not reviews:
        print(f"    [!] Khong co review nao de luu")
        return None
    
    # Tao thu muc output neu chua co
    os.makedirs(output_dir, exist_ok=True)
    
    # Tao DataFrame
    df = pd.json_normalize(reviews)
    
    # Xu ly thoi gian
    if 'reviewTime' in df.columns:
        df['reviewTime'] = pd.to_datetime(df['reviewTime'], unit='ms', errors='coerce')
    
    # Thong ke rating trong data
    if 'rating' in df.columns:
        print(f"\n    Phan bo rating trong file:")
        rating_counts = df['rating'].value_counts().sort_index()
        for r, c in rating_counts.items():
            print(f"      {int(r)} sao: {c}")
    
    # Ten file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"lazada_{item_id}_{len(reviews)}_reviews_balanced"
    excel_file = os.path.join(output_dir, f"{base_name}.xlsx")
    json_file = os.path.join(output_dir, f"{base_name}.json")
    
    # Luu Excel
    df.to_excel(excel_file, index=False)
    print(f"\n    [OK] Da luu Excel: {excel_file}")
    
    # Luu JSON
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)
    print(f"    [OK] Da luu JSON: {json_file}")
    
    return df


def load_urls(url_file):
    """Doc danh sach URL tu file"""
    if not os.path.exists(url_file):
        print(f"[X] Khong tim thay file: {url_file}")
        return []
    
    with open(url_file, 'r', encoding='utf-8') as f:
        urls = [line.strip().strip('"').strip("'") for line in f if line.strip()]
    
    print(f"[OK] Da doc {len(urls)} URLs tu {url_file}")
    return urls


def main():
    print("\n" + "="*60)
    print("    LAZADA REVIEW CRAWLER - BALANCED RATINGS")
    print("="*60)
    print(f"    Thoi gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    Target moi rating: {REVIEWS_PER_RATING}")
    print("="*60)
    
    # Load cookies
    cookie_jar = load_cookies(COOKIE_FILE)
    if not cookie_jar:
        print("\n[X] Khong the tiep tuc do thieu cookie!")
        print("    Hay export cookie tu browser va luu vao:", COOKIE_FILE)
        return
    
    # Tao session
    session = create_session(cookie_jar)
    
    # Load URLs
    urls = load_urls(URL_FILE)
    if not urls:
        # Neu khong co file URL, cho phep nhap thu cong
        print("\n[i] Nhap URL san pham Lazada (hoac 'q' de thoat):")
        while True:
            url = input("URL: ").strip()
            if url.lower() == 'q':
                break
            if url:
                urls.append(url)
    
    if not urls:
        print("[X] Khong co URL nao de cao!")
        return
    
    # Thong ke
    total_reviews = 0
    success_count = 0
    failed_count = 0
    
    # Bat dau cao
    print(f"\n[>] Bat dau cao {len(urls)} san pham...")
    
    for i, url in enumerate(urls, 1):
        print(f"\n{'#'*60}")
        print(f"# SAN PHAM {i}/{len(urls)}")
        print(f"{'#'*60}")
        
        reviews, item_id = crawl_lazada_reviews_balanced(
            session, url,
            reviews_per_rating=REVIEWS_PER_RATING,
            delay_min=DELAY_MIN,
            delay_max=DELAY_MAX
        )
        
        if reviews:
            save_reviews(reviews, item_id, OUTPUT_DIR)
            total_reviews += len(reviews)
            success_count += 1
        else:
            failed_count += 1
        
        # Delay giua cac san pham
        if i < len(urls):
            delay = random.uniform(3, 5)
            print(f"\n    [i] Doi {delay:.1f}s truoc khi cao san pham tiep theo...")
            time.sleep(delay)
    
    # Tong ket
    print("\n" + "="*60)
    print("    HOAN TAT!")
    print("="*60)
    print(f"    Tong san pham: {len(urls)}")
    print(f"    Thanh cong: {success_count}")
    print(f"    That bai: {failed_count}")
    print(f"    Tong reviews: {total_reviews}")
    print(f"    Ket qua luu tai: {OUTPUT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()

# 📜 SCRIPT BÁO CÁO ĐỒ ÁN MÔN HỌC SE363

## **HỆ THỐNG PHÂN TÍCH CẢM XÚC THEO KHÍA CẠNH THỜI GIAN THỰC CHO ĐÁNH GIÁ THƯƠNG MẠI ĐIỆN TỬ**

### (Real-Time Aspect-Based Sentiment Analysis for E-Commerce Reviews)

---

## 🎤 PHẦN MỞ ĐẦU (Slide 1-3) - Khoảng 2 phút

### Slide 1: Giới thiệu

> "Kính chào Thầy/Cô và các bạn. Hôm nay, nhóm em xin được trình bày đồ án môn SE363 - Phát triển ứng dụng trên nền tảng Dữ liệu lớn, với đề tài **Hệ thống Phân tích Cảm xúc theo Khía cạnh Thời gian Thực cho Đánh giá Thương mại Điện tử**."

### Slide 2: Thành viên nhóm

> "Nhóm em gồm có [liệt kê tên thành viên]. Giảng viên hướng dẫn là [Tên giảng viên]."

### Slide 3: Mục lục

> "Bài báo cáo của chúng em sẽ gồm 6 phần chính:
>
> 1. Tổng quan đề tài - đặt vấn đề và mục tiêu
> 2. Cơ sở lý thuyết - ABSA, PhoBERT và công nghệ Big Data
> 3. Thiết kế hệ thống - kiến trúc tổng thể
> 4. Chi tiết hiện thực - các kỹ thuật nổi bật
> 5. Thực nghiệm và đánh giá
> 6. Kết luận và hướng phát triển"

---

## 📌 CHƯƠNG 1: TỔNG QUAN ĐỀ TÀI (Slide 4-8) - Khoảng 4 phút

### Slide 4: Đặt vấn đề

> "Trong kỷ nguyên thương mại điện tử bùng nổ hiện nay, các nền tảng như Lazada, Shopee, Tiki đã trở thành kênh mua sắm chính yếu tại Việt Nam. Một sản phẩm phổ biến có thể nhận được **hàng chục nghìn** lượt đánh giá từ người dùng."
>
> "Tuy nhiên, việc phân tích thủ công lượng dữ liệu khổng lồ này là **bất khả thi**. Các hệ thống Sentiment Analysis truyền thống thường chỉ đưa ra kết quả **nhị phân** - Tích cực hoặc Tiêu cực - cho toàn bộ câu."
>
> "Điều này **không phản ánh đúng thực tế**. Ví dụ, câu đánh giá: _'Sản phẩm rất đẹp nhưng giao hàng quá chậm'_ - đây là câu chứa cả cảm xúc **tích cực** về sản phẩm và **tiêu cực** về vận chuyển. Hệ thống truyền thống không thể phân tách được điều này."

### Slide 5: Mục tiêu đề tài

> "Từ đó, nhóm em xác định 4 mục tiêu chính:
>
> **Thứ nhất**: Xây dựng Pipeline Big Data - tích hợp Apache Kafka và Apache Spark để xử lý luồng dữ liệu streaming tốc độ cao.
>
> **Thứ hai**: Phát triển Mô hình Deep Learning cho tiếng Việt - Fine-tune PhoBERT để giải quyết bài toán **Multi-Polarity ABSA** cho 9 khía cạnh đặc thù của E-commerce Việt Nam.
>
> **Thứ ba**: Tối ưu hóa hiệu năng - đảm bảo độ trễ dưới 5 giây từ lúc thu thập đến khi hiển thị kết quả.
>
> **Thứ tư**: Trực quan hóa - xây dựng Dashboard tương tác giúp người dùng dễ dàng theo dõi và ra quyết định."

### Slide 6: Phạm vi nghiên cứu

> "Về phạm vi:
>
> - **Dữ liệu**: Đánh giá sản phẩm tiếng Việt trên sàn Lazada.vn
> - **Aspects**: 9 khía cạnh chính bao gồm: Chất lượng sản phẩm, Giá cả, Vận chuyển, Đóng gói, Thái độ phục vụ, Tính xác thực, Đúng mô tả, Hiệu năng/Trải nghiệm, và Bảo hành/Đổi trả
> - **Công nghệ**: Hệ sinh thái Python, Docker, Apache Kafka, Apache Spark, HuggingFace Transformers"

### Slide 7-8: Ý nghĩa thực tiễn

> "Hệ thống mang lại giá trị cho cả 3 bên tham gia E-commerce:
>
> **Đối với Người bán (Sellers)**: Nhanh chóng phát hiện vấn đề - ví dụ lô hàng bị lỗi, đơn vị vận chuyển làm hỏng hàng - để khắc phục kịp thời.
>
> **Đối với Người mua (Buyers)**: Có cái nhìn tổng quan trung thực về sản phẩm thay vì phải đọc hàng trăm reviews.
>
> **Đối với Sàn TMĐT**: Giám sát chất lượng nhà bán hàng và xu hướng thị trường."

---

## 📚 CHƯƠNG 2: CƠ SỞ LÝ THUYẾT (Slide 9-16) - Khoảng 5 phút

### Slide 9: Aspect-Based Sentiment Analysis (ABSA)

> "Để mọi người hiểu rõ hơn về nền tảng lý thuyết, em xin trình bày về ABSA.
>
> ABSA - hay Aspect-Based Sentiment Analysis - là bài toán **con** của Sentiment Analysis, nhưng phức tạp hơn. Thay vì chỉ đưa ra một kết luận cho cả câu, ABSA tập trung vào việc **xác định cảm xúc đối với từng khía cạnh** cụ thể.
>
> Trong đề tài này, bài toán được định nghĩa là **Multi-label Multi-class Classification**:
>
> - **Input**: Một câu review X
> - **Output**: Một tập hợp các cặp (Aspect, Polarity) với Polarity có thể là Positive, Negative, Neutral, hoặc None (không đề cập)
>
> Điểm **đặc biệt** trong cách tiếp cận của nhóm là hỗ trợ **Multi-polarity** - tức là một khía cạnh có thể mang cả hai sắc thái đối lập trong cùng một ngữ cảnh. Ví dụ: 'Hàng đẹp nhưng chất lượng kém' - cùng nói về sản phẩm nhưng vừa khen vừa chê."

### Slide 10-11: BERT Architecture

> "Để giải quyết bài toán này, nhóm em sử dụng mô hình **BERT** - Bidirectional Encoder Representations from Transformers.
>
> Sức mạnh của BERT nằm ở cơ chế **Self-Attention**, cho phép mô hình 'nhìn' vào toàn bộ câu cùng một lúc và hiểu ngữ cảnh của từ dựa trên các từ xung quanh - cả bên trái **và** bên phải.
>
> Công thức Self-Attention là: **Attention(Q,K,V) = softmax(QK^T / √d_k) × V**
>
> Trong đó Q, K, V là các ma trận Query, Key, Value được học trong quá trình huấn luyện."

### Slide 12: PhoBERT - BERT cho tiếng Việt

> "Tuy nhiên, BERT gốc không hiểu tốt tiếng Việt. Do đó, nhóm em sử dụng **PhoBERT** - State-of-the-art Language Model cho tiếng Việt, được huấn luyện trên tập dữ liệu **20GB văn bản**.
>
> Khác với Multilingual BERT, PhoBERT sử dụng **tokenizer chuyên biệt** cho tiếng Việt, dựa trên kỹ thuật BPE kết hợp với **word segmentation**. Điều này giúp PhoBERT hiểu rõ hơn về ngữ pháp và từ vựng tiếng Việt.
>
> Trong đồ án, nhóm sử dụng `vinai/phobert-base` làm backbone để trích xuất đặc trưng trước khi đưa vào các lớp phân loại."

### Slide 13-14: Apache Kafka

> "Tiếp theo là phần công nghệ Big Data. Đầu tiên là **Apache Kafka** - nền tảng phân phối sự kiện phân tán.
>
> Kafka có 4 thành phần chính:
>
> - **Producer**: Đẩy dữ liệu vào hệ thống
> - **Topic**: Kênh chứa dữ liệu, ví dụ topic 'raw_reviews'
> - **Consumer**: Đọc dữ liệu để xử lý
> - **Broker**: Server lưu trữ và quản lý Kafka
>
> **Vai trò trong hệ thống**: Kafka đóng vai trò là **bộ đệm tin cậy** (buffer), giúp **tách rời** (decouple) tốc độ thu thập dữ liệu và tốc độ xử lý. Khi lượng review tăng đột biến, Kafka đảm bảo hệ thống không bị quá tải - đây gọi là **backpressure handling**."

### Slide 15: Apache Spark Streaming

> "Thành phần tiếp theo là **Apache Spark** - framework xử lý dữ liệu phân tán.
>
> Nhóm sử dụng **Structured Streaming** - module xử lý luồng của Spark, cho phép viết code xử lý streaming giống như xử lý batch thông qua DataFrame API.
>
> Spark sử dụng **Micro-batch Architecture** - xử lý dữ liệu theo từng batch nhỏ, ví dụ mỗi 1 giây. Điều này giúp **cân bằng** giữa độ trễ (latency) và thông lượng (throughput).
>
> **Vai trò**: Spark nhận dữ liệu từ Kafka, batching lại và gửi vào mô hình Deep Learning để dự đoán song song. Dù hiện tại chạy local mode 1 worker, kiến trúc này **sẵn sàng scale** lên cluster."

### Slide 16: Apache Airflow

> "Cuối cùng là **Apache Airflow** - nền tảng lập lịch và giám sát workflows.
>
> Airflow sử dụng **DAG** - Directed Acyclic Graph - để định nghĩa chuỗi các task phụ thuộc nhau, cùng với các **Operators** như PythonOperator, BashOperator.
>
> **Vai trò**: Airflow điều phối **toàn bộ quy trình**: Kích hoạt Crawler → Kiểm tra Kafka → Kích hoạt Spark Job → Tổng hợp kết quả → Cập nhật Dashboard."

---

## 🏗️ CHƯƠNG 3: THIẾT KẾ HỆ THỐNG (Slide 17-22) - Khoảng 4 phút

### Slide 17-18: Kiến trúc tổng thể

> "Giờ em xin trình bày về thiết kế hệ thống. Hệ thống được thiết kế theo mô hình **Lambda Architecture** giản lược, tập trung vào Speed Layer, bao gồm **7 tầng**:
>
> **Tầng 1 - Data Acquisition Layer**: Gồm Lazada Crawler sử dụng Python/Requests/Selenium để thu thập review, xác thực và deduplication.
>
> **Tầng 2 - Messaging Layer**: Apache Kafka + Zookeeper, đóng vai trò Message Broker, đảm bảo độ tin cậy và thứ tự dữ liệu.
>
> **Tầng 3 - Streaming Processing Layer**: Apache Spark PySpark nhận data từ Kafka và micro-batching.
>
> **Tầng 4 - AI Inference Layer**: PhoBERT Model chạy trên PyTorch để dự đoán cảm xúc từ văn bản thô.
>
> **Tầng 5 - Persistence Layer**: JSON Files cho Local Storage, PostgreSQL cho Airflow Metadata.
>
> **Tầng 6 - Visualization Layer**: Streamlit Dashboard hiển thị biểu đồ và tương tác người dùng.
>
> **Tầng 7 - Orchestration Layer**: Airflow quản lý luồng công việc tự động."

### Slide 19-20: Data Flow Diagram

> "Về luồng dữ liệu: Review từ Lazada → Crawler thu thập → Producer đẩy vào Kafka topic 'raw_reviews' → Spark Consumer đọc và batching → PhoBERT Predictor inference → Kết quả ghi vào JSON → Dashboard hiển thị.
>
> Toàn bộ quy trình này được Airflow điều phối tự động theo lịch định sẵn."

### Slide 21-22: Database Schema

> "Về thiết kế dữ liệu, nhóm sử dụng JSON cho tính linh hoạt của NoSQL nhưng schema được quy định chặt chẽ.
>
> **Raw Data Schema** từ Kafka gồm: review_id, content, rating (1-5 sao), timestamp, và item_id.
>
> **Processed Data Schema** sau inference gồm: product_id, mảng reviews chứa các predictions với giá trị aspect (1=Positive, -1=Negative, 0=Neutral)."

---

## 💻 CHƯƠNG 4: CHI TIẾT HIỆN THỰC (Slide 23-38) - Khoảng 8 phút

> "Đây là chương **trọng tâm** mô tả các kỹ thuật chuyên sâu mà nhóm đã áp dụng."

### Slide 23-25: Balanced Sampling

> "**Kỹ thuật đầu tiên: Balanced Sampling**
>
> Vấn đề: Dữ liệu review trên E-commerce bị **lệch rất nặng** về phía tích cực - 5 sao chiếm tới 70-80%. Nếu crawl ngẫu nhiên, mô hình sẽ **không có đủ negative samples** để học.
>
> **Giải pháp của nhóm**: Thay vì request API mặc định, nhóm thực hiện các request **riêng biệt** cho từng filter rating. Cụ thể:
>
> ```python
> REVIEWS_PER_RATING = {1: 30, 2: 30, 3: 30, 4: 30, 5: 30}
> for rating in [1, 2, 3, 4, 5]:
>     params = {'filter': str(rating), ...}
>     batch = fetch_reviews(params)
> ```
>
> **Kết quả**: Dataset thu được có **phân phối đồng đều**, giúp mô hình học cân bằng giữa các lớp cảm xúc."

### Slide 26-27: Cookie Authentication

> "**Kỹ thuật thứ hai: Hybrid Cookie Injection**
>
> Lazada sử dụng cơ chế **Anti-bot mạnh mẽ**. Để vượt qua, nhóm sử dụng phương pháp Hybrid với 4 bước:
>
> - **Bước 1**: Sử dụng Selenium Browser để người dùng đăng nhập thủ công an toàn
> - **Bước 2**: Dump cookies từ browser session ra file định dạng Netscape 'cookies.txt'
> - **Bước 3**: Inject cookies này vào requests.Session() của Crawler headless
> - **Bước 4**: Giả lập Headers (User-Agent, Sec-Ch-Ua, Referer) giống hệt trình duyệt thật
>
> Điều này cho phép crawler hoạt động **nhanh như API** nhưng vẫn **bypass được bảo vệ**."

### Slide 28-29: Fuzzy Deduplication

> "**Kỹ thuật thứ ba: Fuzzy Deduplication**
>
> Trong môi trường Big Data, trùng lặp không chỉ là match 100% (exact match) mà còn là trùng lặp do **spam bot** - thêm dấu chấm, dấu cách.
>
> Nhóm sử dụng kỹ thuật **Canonical Signature**:
>
> 1. Lowercase toàn bộ văn bản
> 2. Loại bỏ toàn bộ punctuation và whitespace
> 3. Tạo hash làm key để deduplicate
>
> Ví dụ: 'Hàng đẹp quá !!!' và 'Hàng đẹp quá.' đều cho ra signature 'hangdepqua' → Hệ thống phát hiện trùng lặp và loại bỏ."

### Slide 30-31: Kafka Partitioning Strategy

> "**Kỹ thuật thứ tư: Kafka Partitioning Strategy**
>
> Thách thức của Distributed System là **thứ tự** (Ordering). Làm sao đảm bảo các review của cùng một sản phẩm được xử lý đúng?
>
> **Giải pháp**: Sử dụng `Product_ID` làm Partition Key:
>
> ```python
> producer.send(topic='raw_reviews', key=product_id.encode('utf-8'), value=json.dumps(review))
> ```
>
> Kafka đảm bảo tất cả message có **cùng key** sẽ luôn được đẩy vào **cùng một Partition**. Điều này cực kỳ quan trọng cho Spark consumer có thể aggregate dữ liệu theo sản phẩm **hiệu quả** mà không cần shuffle nhiều."

### Slide 32-34: Pandas UDF Optimization

> "**Kỹ thuật thứ năm: Pandas UDF - đây là kỹ thuật quan trọng nhất!**
>
> Cách ngây thơ (Naive approach) là loop qua từng row của Spark DataFrame và gọi model predict. Điều này **cực chậm** vì overhead của việc chuyển đổi dữ liệu giữa JVM (Java) và Python.
>
> **Giải pháp của nhóm: Scalar Iterator Pandas UDF**
>
> ```python
> @pandas_udf(StringType())
> def predict_batch_udf(content_series: pd.Series) -> pd.Series:
>     global model
>     if model is None: model = load_model()  # Load once per partition
>     predictions = model.predict(content_series.tolist())  # Batch inference
>     return pd.Series([json.dumps(p) for p in predictions])
> ```
>
> **Lợi ích**:
>
> - **Zero-copy**: Sử dụng Apache Arrow để truyền dữ liệu giữa JVM và Python
> - **Batch Inference**: Predict một lúc 100-200 samples thay vì từng cái một - tận dụng GPU vectorization"

### Slide 35-36: Multi-Task Learning Architecture & Multi-Polarity (TRỌNG TÂM)

> "**Kỹ thuật thứ sáu: Multi-Polarity ABSA - Đây là điểm ĐẶC BIỆT NHẤT của hệ thống!**"

---

#### 🔴 Vấn đề với các hệ thống truyền thống

> "Các hệ thống ABSA truyền thống sử dụng **Single-label Classification** cho sentiment - tức mỗi khía cạnh chỉ có **DUY NHẤT một nhãn**: Positive, Negative, hoặc Neutral.
>
> Tuy nhiên trong thực tế, một bình luận có thể chứa **CẢ KHEN VÀ CHÊ** cho **CÙNG MỘT khía cạnh**!
>
> **Ví dụ thực tế:**
>
> - _'Áo đẹp nhưng vải hơi mỏng'_ → Cùng nói về **Chất lượng sản phẩm**, vừa KHEN (đẹp) vừa CHÊ (mỏng)
> - _'Sản phẩm chất lượng tốt cho giá tiền này, form hơi rộng một chút'_ → **Chất lượng** vừa POS vừa NEG
> - _'Giá rẻ, chất ổn nhưng size hơi lớn'_ → **Chất lượng** vừa POS (ổn) vừa NEG (size lớn)
>
> Nếu dùng single-label, hệ thống **buộc phải chọn 1 trong 2**, dẫn đến **mất thông tin quan trọng**."

---

#### 🟢 Giải pháp: Multi-Polarity Classification

> "Nhóm em đề xuất chuyển bài toán Sentiment Classification từ **Multi-class** (chọn 1 trong 3) sang **Multi-label** (có thể chọn nhiều cùng lúc).
>
> **Định dạng nhãn Multi-Polarity:**
>
> | Giá trị   | Ý nghĩa                                 |
> | --------- | --------------------------------------- |
> | `1`       | 😊 Tích cực (Positive)                  |
> | `0`       | 😐 Trung lập (Neutral)                  |
> | `-1`      | 😞 Tiêu cực (Negative)                  |
> | `2`       | ❌ Không nhắc đến                       |
> | `[-1, 1]` | 🔀 **ĐA CỰC**: Vừa tiêu cực VÀ tích cực |
>
> Ví dụ với câu _'Áo đẹp nhưng vải hơi mỏng'_:
>
> - Khía cạnh **Chất lượng sản phẩm**: được gán nhãn `[-1, 1]` thay vì chỉ `1` hoặc `-1`"

---

#### 🏗️ Kiến trúc mô hình Multi-Polarity

> "Mô hình được thiết kế theo kiến trúc **Multi-Task Learning** với **Hard Parameter Sharing**:
>
> ```
> Input Text
>     ↓
> [PhoBERT Encoder] ← Shared backbone (768 hidden dimensions)
>     ↓
> [CLS] Token Representation
>     ↓
> ┌────────────────┬─────────────────────┐
> │   HEAD 1       │      HEAD 2         │
> │ Mention Detect │ Sentiment Classify  │
> │ (Binary/aspect)│ (Multi-label/aspect)│
> └────────────────┴─────────────────────┘
>     ↓                    ↓
> [9 neurons]      [9 × 3 neurons]
> Sigmoid          Sigmoid (NOT Softmax!)
> ```
>
> **Điểm khác biệt quan trọng:**
>
> - HEAD 1 (**Mention Detection**): Output `[batch_size, 9_aspects]` - Binary classification: khía cạnh này có được nhắc đến không?
> - HEAD 2 (**Sentiment Classification**): Output `[batch_size, 9_aspects, 3_polarities]` - Với mỗi khía cạnh, cho biết xác suất của từng loại sentiment (NEG, POS, NEU)
>
> **Tại sao dùng Sigmoid thay vì Softmax?**
>
> - Softmax **buộc các class cạnh tranh nhau** - tổng xác suất = 1, nên nếu POS tăng thì NEG phải giảm
> - Sigmoid cho phép **mỗi class độc lập** - POS và NEG có thể CÙNG CAO nếu review vừa khen vừa chê"

---

#### 📐 Loss Function

> "Loss Function tổng hợp cho Multi-Task Learning:
>
> **Total Loss = Loss_Mention + Loss_Sentiment**
>
> Trong đó:
>
> - `Loss_Mention` = **BCEWithLogitsLoss** cho bài toán mention detection (binary)
> - `Loss_Sentiment` = **BCEWithLogitsLoss** cho bài toán sentiment (multi-label) ← THAY ĐỔI so với CrossEntropyLoss truyền thống!
>
> Code trong `phobert_trainer_multipolarity.py`:
>
> ````python
> # Loss functions - BOTH use BCE for multi-label!
> criterion_m = nn.BCEWithLogitsLoss()  # Binary cho mention detection
> criterion_s = nn.BCEWithLogitsLoss()  # Multi-label cho sentiment (CHANGED!)
>
> # Predictions - BOTH use sigmoid threshold
> preds_m = (torch.sigmoid(logits_m) > 0.5).float()
> preds_s = (torch.sigmoid(logits_s) > 0.5).float()  # Multi-label!
> ```"
> ````

---

#### 📊 Minh họa với ví dụ cụ thể

> "**Ví dụ chi tiết với review thực tế:**
>
> Input: _'Áo đẹp nhưng vải hơi mỏng, ship nhanh, đóng gói cẩn thận'_
>
> **Bước 1 - Mention Detection:** (aspect nào được nhắc đến?)
>
> | Khía cạnh           | Mention                           |
> | ------------------- | --------------------------------- |
> | Chất lượng sản phẩm | ✅ 1 (có nhắc: áo đẹp, vải mỏng)  |
> | Vận chuyển          | ✅ 1 (có nhắc: ship nhanh)        |
> | Đóng gói            | ✅ 1 (có nhắc: đóng gói cẩn thận) |
> | Giá cả              | ❌ 0                              |
> | ...                 | ❌ 0                              |
>
> **Bước 2 - Sentiment Classification:** (cảm xúc là gì?)
>
> | Khía cạnh  | NEG         | POS        | NEU  | Kết quả                     |
> | ---------- | ----------- | ---------- | ---- | --------------------------- |
> | Chất lượng | ✅ 1 (mỏng) | ✅ 1 (đẹp) | ❌ 0 | **[-1, 1]** Multi-polarity! |
> | Vận chuyển | ❌ 0        | ✅ 1       | ❌ 0 | **1** (Positive)            |
> | Đóng gói   | ❌ 0        | ✅ 1       | ❌ 0 | **1** (Positive)            |
>
> Như vậy, hệ thống **KHÔNG mất thông tin** rằng khách hàng vừa thích (đẹp) vừa không thích (mỏng) về chất lượng!"

---

#### 🎯 Ý nghĩa thực tiễn

> "**Tại sao Multi-Polarity quan trọng cho E-commerce?**
>
> 1. **Phản ánh đúng thực tế**: Người mua thường có ý kiến trái chiều trong cùng 1 review
> 2. **Insight chi tiết hơn**: Seller biết được cụ thể điểm nào được khen, điểm nào bị chê của cùng một khía cạnh
> 3. **Không mất thông tin**: Thay vì phải chọn 1 label, hệ thống giữ lại TẤT CẢ thông tin sentiment
>
> **Thống kê từ dữ liệu thực tế:**
>
> - Trong tập train 2,500 reviews, có khoảng **5-10%** mẫu có multi-polarity
> - Tỷ lệ này tuy nhỏ nhưng chứa thông tin CỰC KỲ GIÁ TRỊ cho phân tích!"

### Slide 37-38: Atomic File Writing

> "**Kỹ thuật cuối cùng: Atomic File Writing**
>
> Vấn đề: Khi Dashboard đọc file JSON **cùng lúc** Spark đang ghi, lỗi JSONDecodeError hoặc partial read sẽ xảy ra.
>
> **Giải pháp Atomic Write**:
>
> ```python
> # Ghi vào file tạm
> with open(temp_path, 'w') as f:
>     json.dump(data, f)
> # Rename file (Hệ điều hành đảm bảo thao tác này là atomic)
> os.replace(temp_path, final_path)
> ```
>
> Điều này đảm bảo Dashboard **hoặc** đọc được file cũ, **hoặc** đọc được file mới hoàn chỉnh, **không bao giờ** đọc phải file lỗi."

---

## 📊 CHƯƠNG 5: THỰC NGHIỆM VÀ ĐÁNH GIÁ (Slide 39-45) - Khoảng 4 phút

### Slide 39: Mô tả tập dữ liệu

> "Về thực nghiệm, tập dữ liệu huấn luyện bao gồm **2,500 đánh giá** được gán nhãn thủ công (Manual Annotation).
>
> Quy trình gán nhãn tuân thủ nghiêm ngặt Annotation Guidelines đã xây dựng, với độ đo **Inter-annotator agreement** đạt Kappa > 0.8 - cho thấy độ nhất quán cao giữa các annotators."

### Slide 40-41: Model Evaluation

> "Nhóm sử dụng **5-Fold Cross Validation** để đánh giá mô hình.
>
> Kết quả trung bình trên Test set:
>
> - **Chất lượng sản phẩm**: F1 = 0.885
> - **Giao hàng**: F1 = 0.910 (cao nhất!)
> - **Giá cả**: F1 = 0.845
> - **Macro Average**: **F1 = 0.865**
>
> **Nhận xét**: Mô hình hoạt động **rất tốt** trên các khía cạnh phổ biến (Chất lượng, Giao hàng). Các khía cạnh ít dữ liệu hơn như Tính xác thực có độ chính xác thấp hơn đôi chút, cần bổ sung thêm dữ liệu."

### Slide 42-43: System Performance

> "Về hiệu năng hệ thống, thử nghiệm trên cấu hình: CPU Intel Core i7, RAM 16GB, GPU NVIDIA GTX 1650.
>
> **Throughput**: Xử lý trung bình **100 reviews/phút** (bao gồm cả crawling và inference)
>
> **Latency**:
>
> - Crawl → Kafka: ~100ms
> - Kafka → Spark: ~50ms
> - Inference (Spark): ~40ms/review (nhờ Batching)
> - **Tổng độ trễ End-to-End: dưới 5 giây** - đạt mục tiêu đề ra!"

### Slide 44-45: Demo kịch bản

> "Nhóm xin demo 2 kịch bản sử dụng:
>
> **Kịch bản 1: Sản phẩm có vấn đề vận chuyển**
>
> - Input: URL một sản phẩm tai nghe giá rẻ
> - Dashboard hiển thị: Rating tổng quan **4.5 sao** (cao), nhưng cột Aspect 'Vận chuyển' **đỏ rực** (Negative chiếm 80%)
> - **Kết luận**: Sản phẩm tốt nhưng đơn vị vận chuyển làm móp hộp. Seller cần đổi đối tác vận chuyển hoặc đóng gói kỹ hơn.
>
> **Kịch bản 2: Sản phẩm giả mạo**
>
> - Input: URL một sản phẩm mỹ phẩm luxury giá rẻ bất thường
> - Dashboard: Aspect 'Tính xác thực' và 'Đúng mô tả' có nhiều **cảnh báo Negative**
> - **Kết luận**: Hệ thống cảnh báo người mua tiềm năng về rủi ro hàng giả."

---

## 🎯 CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (Slide 46-48) - Khoảng 2 phút

### Slide 46: Kết luận

> "Đồ án đã hoàn thành **xuất sắc** các mục tiêu đề ra:
>
> 1. Đã xây dựng thành công **pipeline xử lý dữ liệu lớn thời gian thực** với Kafka + Spark
> 2. Áp dụng hiệu quả các kỹ thuật **State-of-the-art** vào bài toán thực tế - PhoBERT, Spark Streaming, Pandas UDF
> 3. Hệ thống chạy **ổn định**, xử lý được các edge cases như mất mạng, captcha, dữ liệu bẩn"

### Slide 47: Hướng phát triển

> "Về hướng phát triển trong tương lai, nhóm đề xuất 3 hướng:
>
> **1. Mở rộng nguồn dữ liệu**: Tích hợp thêm Shopee, Tiki, Facebook Comments
>
> **2. Auto-scaling**: Triển khai Spark và Kafka trên Kubernetes (K8s) để tự động mở rộng khi tải tăng cao
>
> **3. Active Learning**: Xây dựng cơ chế cho phép người dùng sửa nhãn sai trên Dashboard và model **tự động học lại** (Retrain) định kỳ"

### Slide 48: Kết thúc

> "Nhóm em xin chân thành cảm ơn Thầy/Cô đã lắng nghe. Chúng em sẵn sàng giải đáp mọi câu hỏi."

---

## ❓ CÂU HỎI DỰ KIẾN VÀ TRẢ LỜI

### Q1: Tại sao chọn PhoBERT thay vì các mô hình khác?

> "PhoBERT được huấn luyện trên 20GB văn bản tiếng Việt, sử dụng tokenizer chuyên biệt với word segmentation. So với Multilingual BERT, PhoBERT hiểu ngữ pháp và từ vựng tiếng Việt tốt hơn nhiều. Benchmark cho thấy PhoBERT đạt SOTA trên hầu hết các task NLP tiếng Việt."

### Q2: Làm sao xử lý khi Lazada block crawler?

> "Nhóm sử dụng Hybrid Cookie Injection - đăng nhập thủ công qua Selenium, export cookies rồi inject vào headless requests. Kết hợp với giả lập headers giống trình duyệt thật, random delay giữa các request. Nếu vẫn bị block, hệ thống có fallback mechanism retry với exponential backoff."

### Q3: Tại sao dùng Kafka thay vì RabbitMQ?

> "Kafka được thiết kế cho high-throughput và durability. Messages trong Kafka được persist trên disk, có thể replay. Kafka cũng hỗ trợ partitioning cho parallel processing. RabbitMQ phù hợp hơn cho task queue với complex routing, còn với streaming data như reviews thì Kafka là lựa chọn tốt hơn."

### Q4: Multi-polarity là gì và tại sao cần thiết?

> "Multi-polarity cho phép một khía cạnh mang cả sentiment tích cực VÀ tiêu cực. Ví dụ: 'Hàng đẹp nhưng chất lượng kém' - cùng về sản phẩm nhưng vừa khen (đẹp) vừa chê (kém chất lượng). Hệ thống truyền thống chỉ cho 1 label sẽ bỏ sót thông tin quan trọng."

### Q5: Atomic file writing hoạt động như thế nào?

> "Thay vì ghi trực tiếp vào file đích, nhóm ghi vào file tạm trước, sau đó dùng os.replace() để rename. Hệ điều hành đảm bảo thao tác rename là atomic - hoặc hoàn thành 100% hoặc không thay đổi gì. Điều này tránh tình trạng Dashboard đọc file dở dang."

### Q6: Đánh giá độ chính xác của mô hình như thế nào?

> "Nhóm sử dụng 5-Fold Cross Validation với các metric: Precision, Recall, F1-Score cho từng aspect. Macro Average F1 đạt 0.865. Đồng thời nhóm cũng đo Inter-annotator agreement Kappa > 0.8 để đảm bảo chất lượng dữ liệu huấn luyện."

---

## 📝 LƯU Ý KHI BÁO CÁO

1. **Thời gian**: Ước tính tổng ~25-30 phút. Điều chỉnh tốc độ tùy theo thời gian cho phép.

2. **Giọng nói**: Nói rõ ràng, chậm rãi. Nhấn mạnh các từ khóa kỹ thuật.

3. **Eye contact**: Nhìn vào giảng viên và các bạn, không chỉ nhìn vào slide.

4. **Demo live**: Nếu có thời gian, demo trực tiếp Dashboard với 1-2 sản phẩm thực tế.

5. **Phân công**: Mỗi thành viên trình bày 1-2 chương để thể hiện sự đóng góp.

6. **Chuẩn bị backup**: In sẵn slides ra giấy phòng trường hợp máy tính gặp sự cố.

---

**Chúc nhóm báo cáo thành công! 🎉**

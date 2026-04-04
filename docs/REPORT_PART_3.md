# PART 3: IMPLEMENTATION DETAILS

## CHAPTER 4: CORE IMPLEMENTATION

This chapter provides a deep dive into the source code and the specific algorithms used to implement the system designed in Chapter 3.

### 4.1. Data Acquisition Implementation

The crawler module (`app/crawl_local.py`) is designed to overcome two primary challenges: **Data Imbalance** and **Anti-Bot Protection**.

#### 4.1.1. Balanced Sampling Algorithm

As discussed, e-commerce reviews are naturally skewed towards positive ratings. To create a balanced training dataset for the AI model, we implemented a targeted crawling strategy. Instead of fetching reviews sequentially, the crawler loops through specific rating filters.

**Code Snippet (`app/crawl_local.py`):**

```python
REVIEWS_PER_RATING = {1: 30, 2: 30, 3: 30, 4: 30, 5: 30}

def crawl_lazada_reviews_balanced(session, product_url, reviews_per_rating, ...):
    # Loop through each rating from 1 to 5
    for rating in [1, 2, 3, 4, 5]:
        target = reviews_per_rating.get(rating, 50)

        # Determine specific API endpoint parameters for this rating
        params = {
            "itemId": item_id,
            "filter": str(rating),  # Key parameter: 1=1star, 5=5stars
            "sort": "0"
        }

        # Fetch reviews until target count is reached
        reviews = crawl_reviews_by_rating(session, item_id, rating, target, ...)

        # Deduplication check
        unique_reviews = [r for r in reviews if r.get('review_id') not in seen_ids]
        all_reviews.extend(unique_reviews)
```

This ensures that for every product, we attempt to get an equal distribution of positive and negative feedback, which is crucial for the **Multi-Polarity** model to learn negative nuances.

#### 4.1.2. Hybrid Anti-Bot Bypass

Lazada employs sophisticated anti-crawling measures. We bypass these using a hybrid approach:

1.  **Browser Session**: A Selenium instance logs in manually (handled in `lazada_browser.py`) to generate a valid session.
2.  **Cookie Injection**: The session cookies are exported to a Netscape-formatted file.
3.  **Headless Masquerading**: The Python `requests` session is configured to mimic the exact headers of a Chrome 129 browser on Windows.

**Code Snippet (`app/crawl_local.py`):**

```python
def create_session(cookie_jar):
    session = requests.Session()
    session.cookies = cookie_jar
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ...",
        "Referer": "https://www.lazada.vn/",
        "sec-ch-ua": '"Google Chrome";v="129"...'
    })
    return session
```

### 4.2. Messaging Layer Implementation

The Producer module (`app/lazada_producer.py`) handles the robust ingestion of data into Kafka.

#### 4.2.1. Partitioning Strategy

To ensure that all reviews belonging to a specific product are processed by the same Spark executor (or at least stored contiguously), we use the `product_id` as the message key.

**Code Snippet (`app/lazada_producer.py`):**

```python
producer.send(
    TOPIC_NAME,
    key=product_id.encode('utf-8'),  # Partition Key
    value=message
)
```

Kafka hashes this key to determine the partition. This ordering guarantee is essential for stateful processing if we decide to implement session windows in the future.

### 4.3. Streaming Processing Implementation

The Spark Consumer (`app/kafka_absa_consumer.py`) is the most complex component, integrating JVM-based Spark with Python-based PyTorch.

#### 4.3.1. High-Performance Inference with Pandas UDF

We utilize **Scalar Iterator Pandas UDFs** to perform vectorised inference. Unlike standard Python UDFs which process row-by-row (serializing/deserializing each time), Pandas UDFs transfer data in batches using **Apache Arrow**.

**Code Snippet (`app/kafka_absa_consumer.py`):**

```python
@pandas_udf(StringType())
def predict_model_udf(texts: pd.Series) -> pd.Series:
    # 1. Initialize Singleton Model (Loaded once per Executor)
    if not hasattr(predict_model_udf, 'predictor'):
        predict_model_udf.predictor = PhoBERTPredictor(model_path)

    # 2. Batch Inference
    # The 'texts' input is a pandas Series containing hundreds of reviews
    results = []
    for text in texts:
        preds = predict_model_udf.predictor.predict_single(text)
        results.append(json.dumps(preds))

    return pd.Series(results)
```

#### 4.3.2. Atomic File Writing

To prevent read-write race conditions where the Dashboard attempts to read a JSON file while Spark is still writing to it, we implement **Atomic Writes**.

**Code Snippet (`app/kafka_absa_consumer.py`):**

```python
def save_predictions_atomic(file_path, data):
    temp_path = f"{file_path}.tmp"

    # 1. Write to temporary file
    with open(temp_path, 'w') as f:
        json.dump(data, f)
        os.fsync(f.fileno())  # Force write to disk

    # 2. Atomic Rename
    # This operation is guaranteed by the OS to be atomic
    os.replace(temp_path, file_path)
```

### 4.4. AI Model Implementation

The core intelligence resides in `phobert_trainer_multipolarity.py`.

#### 4.4.1. Multi-Task Model Architecture

We customized the PhoBERT architecture to handle two simultaneous tasks:

1.  **Mention Detection**: Binary classification (Is the aspect mentioned?).
2.  **Sentiment Classification**: Multi-label classification (What are the sentiments?).

**Code Snippet (`phobert_trainer_multipolarity.py`):**

```python
class PhoBERTForABSAMultiPolarity(nn.Module):
    def __init__(self, num_aspects=9):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")

        # Head 1: Mention Detection (9 outputs)
        self.head_m = nn.Linear(768, num_aspects)

        # Head 2: Sentiment Classification (9 * 3 outputs)
        # We output 3 logits (NEG, POS, NEU) for EACH aspect
        self.head_s = nn.Linear(768, num_aspects * 3)

    def forward(self, input_ids, attention_mask):
        # ... (PhoBERT encoding) ...
        # Standard Hard Parameter Sharing
        logits_m = self.head_m(h_cls)
        logits_s = self.head_s(h_cls).view(-1, self.num_aspects, 3)
        return logits_m, logits_s
```

#### 4.4.2. Custom Loss Function for Multi-Polarity

Critically, we changed the loss function for sentiment from `CrossEntropyLoss` (which forces a single exclusive class) to `BCEWithLogitsLoss` (Binary Cross Entropy). This allows the model to predict **both** Positive and Negative for the same aspect if the logits for both classes exceed the threshold.

**Code Snippet (`phobert_trainer_multipolarity.py`):**

```python
# Both tasks use BCE!
criterion_m = nn.BCEWithLogitsLoss()
criterion_s = nn.BCEWithLogitsLoss()

# During Training
loss_m = criterion_m(logits_m, labels_m)
loss_s = criterion_s(logits_s, labels_s) # Multi-hot targets
total_loss = loss_m + loss_s
```

For example, if the input is "Good price but slow shipping", the target vector for 'Shipping' might be `[1, 0, 0]` (Negative), but for "Cheap but fake", the 'Price' aspect could be `[1, 1, 0]` (Positive because cheap, Negative because implies low quality/fake). _Note: In our specific dataset, we simplify "fake" to Authenticity aspect, but the architecture supports this duality._

---

**(End of Part 3)**

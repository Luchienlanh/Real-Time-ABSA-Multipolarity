# PART 2: SYSTEM DESIGN & ARCHITECTURE

## CHAPTER 3: SYSTEM DESIGN

### 3.1. Architectural Pattern: Lambda Architecture (Speed Layer Focus)

In Big Data systems, the **Lambda Architecture** is a popular design pattern that handles massive quantities of data by taking advantage of both batch-processing and stream-processing methods. It normally consists of three layers:

1.  **Batch Layer**: Manages the master dataset (immutable, append-only) and pre-computes batch views.
2.  **Speed Layer**: Processes recent data in real-time to compensate for the high latency of the batch layer.
3.  **Serving Layer**: Indexes the batch views so they can be queried in low-latency.

**Our Adaptation**:
Since the primary requirement of this project is **Real-Time Analysis**, we focus heavily on the **Speed Layer**. We have streamlined the architecture to prioritize low latency. We do not maintain a heavy Batch Layer (like Hadoop HDFS) for historical reprocessing in this iteration, although the system design allows for it to be added later. Instead, we use a lightweight persistence layer (JSON/PostgreSQL) that acts as both a storage for the Speed Layer's output and a source for the serving layer.

### 3.2. Detailed 7-Layer Architecture

The system is decomposed into 7 distinct logical layers, ensuring separation of concerns and maintainability.

#### 3.2.1. Layer 1: Data Acquisition Layer (The Crawler)

- **Responsibility**: To interface with external E-commerce platforms (Lazada) and fetch raw user reviews.
- **Challenges**: Anti-bot protections (Captchas, IP blocks), dynamic content loading (AJAX/React), and messy raw text.
- **Technology**:
  - **Python**: The core language.
  - **Selenium**: For browser automation to handle login and initial cookie acquisition.
  - **Requests**: For high-speed HTTP requests using the injected cookies.
- **Output**: Raw JSON objects containing review text, rating, timestamp, and product ID.

#### 3.2.2. Layer 2: Messaging Layer (The Broker)

- **Responsibility**: To decouple the Crawler from the Processor. It ensures that if the Crawler fetches 1000 reviews/second but the Model can only process 100/second, the system doesn't crash. The extra reviews are safely queued.
- **Technology**: **Apache Kafka** deployed via Docker.
- **Configuration**:
  - **Topic**: `raw_reviews`
  - **Partitions**: Configured to allow parallel consumption (though currently running single-consumer for simplicity).
  - **Replication Factor**: 1 (Local dev) or 3 (Production) for fault tolerance.

#### 3.2.3. Layer 3: Streaming Processing Layer (The Processor)

- **Responsibility**: To read data from Kafka, perform transformation (ETL), and prepare it for the AI model.
- **Technology**: **Apache Spark Structured Streaming** (PySpark).
- **Logic**:
  1.  Read from Kafka stream.
  2.  Parse JSON Schema.
  3.  Group reviews into micro-batches (e.g., 5 seconds window).
  4.  Send batch to Layer 4 for inference.

#### 3.2.4. Layer 4: AI Inference Layer (The Brain)

- **Responsibility**: To understand the Vietnamese text and extract Sentiment-Aspect pairs.
- **Technology**:
  - **PyTorch**: Deep Learning framework.
  - **HuggingFace Transformers**: To load the PhoBERT architecture.
  - **Pandas UDF (User Defined Function)**: To bridge Spark and PyTorch efficiently.
- **Model**: A custom fine-tuned `PhoBERT-base` with Multi-Task classification heads.

#### 3.2.5. Layer 5: Persistence Layer (The Storage)

- **Responsibility**: To store the analyzed results for the Dashboard to consume.
- **Technology**:
  - **JSON Files**: For simple, file-based storage of processed results (`processed_data/`). This was chosen for simplicity in the prototype phase to avoid the overhead of setting up a heavy NoSQL cluster like MongoDB or Cassandra.
  - **PostgreSQL**: Used by Airflow to store task metadata and logs.

#### 3.2.6. Layer 6: Serving & Visualization Layer (The Interface)

- **Responsibility**: To present the data to the end-user (Seller/Buyer) in an understandable format.
- **Technology**: **Streamlit**.
- **Features**:
  - Real-time updates (auto-refresh).
  - Interactive charts (Bar charts for aspect sentiment distribution).
  - Drill-down capability (Click on "Shipping" to typically see specific negative reviews).

#### 3.2.7. Layer 7: Orchestration Layer (The Conductor)

- **Responsibility**: To manage the lifecycle of all other services. It ensures the Crawler starts before the Spark job, checks for failures, and restarts services if they die.
- **Technology**: **Apache Airflow**.

### 3.3. System Data Flow

The data flows through the system in a strictly unidirectional pipeline:

1.  **Trigger**: Airflow triggers the DAG.
2.  **Crawl**: The `LazadaCrawler` logs in, navigates to a product page, and starts fetching reviews.
3.  **Produce**: For every fetched review, it is serialized to JSON and sent to the Kafka topic `raw_reviews`.
4.  **Consume**: Spark Streaming acts as a listener on `raw_reviews`. It picks up new messages as they arrive.
5.  **Inference**:
    - Spark accumulates a micro-batch (e.g., 50 reviews).
    - It passes this batch to the PhoBERT model.
    - The Model outputs a prediction tensor.
    - Spark converts this tensor back to structured data (Aspect: "Quality", Sentiment: "Positive").
6.  **Store**: Spark writes the result to a JSON file in the `data/processed` directory.
7.  **Visualize**: The Streamlit Dashboard watches the `data/processed` directory. When the file changes, it re-renders the charts to reflect the new data.

### 3.4. Database Design & Schema

#### 3.4.1. Constraints and Requirements

- **Flexibility**: Review structures can change (e.g., Lazada adds a "video review" field).
- **Read Speed**: The Dashboard needs to read the entire history of a product quickly.
- **Write Speed**: Spark needs to append high-velocity data.

Given these requirements, a **Document-Oriented** approach (JSON) was selected.

#### 3.4.2. Raw Data Schema (Kafka)

This is the payload sent from Layer 1 to Layer 2.

| Field Name  | Data Type | Description                               |
| ----------- | --------- | ----------------------------------------- |
| `review_id` | String    | Unique ID from Lazada (deduplication key) |
| `item_id`   | String    | ID of the product being reviewed          |
| `rating`    | Integer   | User's star rating (1-5)                  |
| `content`   | String    | The actual text of the review             |
| `timestamp` | Long      | Unix timestamp of crawl time              |

**Example**:

```json
{
  "review_id": "12345678",
  "item_id": "9999",
  "rating": 5,
  "content": "Giao hàng nhanh, máy đẹp.",
  "timestamp": 1704067200
}
```

#### 3.4.3. Processed Data Schema (Storage)

This is the output from Layer 4 stored in Layer 5.

| Field Name          | Data Type     | Description               |
| ------------------- | ------------- | ------------------------- |
| `product_id`        | String        | The product ID            |
| `total_reviews`     | Integer       | Counter for aggregation   |
| `reviews`           | Array[Object] | List of analyzed reviews  |
| `reviews.sentiment` | Object        | Map of Aspect -> Polarity |

**Example**:

```json
{
  "product_id": "9999",
  "reviews": [
    {
      "original_content": "Giao hàng nhanh, máy đẹp.",
      "sentiment": {
        "shipping": 1,
        "quality": 1,
        "price": 0
      }
    }
    // ... more reviews
  ]
}
```

Note on Polarity Encoding:

- `1`: Positive
- `-1`: Negative
- `0`: Neutral
- (Missing key): Aspect not mentioned.

### 3.5. Technology Selection Justification

#### Why Kafka and not RabbitMQ?

RabbitMQ is a traditional message queue (smart broker, dumb consumer) great for complex routing. Kafka is an event streaming platform (dumb broker, smart consumer) designed for **high throughput**. Since e-commerce reviews can arrive in bursts (e.g., during a Flash Sale), Kafka's ability to persist logs to disk and handle massive throughput makes it superior for this specific use case.

#### Why Spark Structured Streaming?

We chose Structured Streaming over the legacy Spark Streaming (DStream) because it operates on **DataFrames**. This allows us to use the same SQL-like optimizations for streaming as we do for batch processing. Furthermore, its integration with **Pandas UDF** allows us to run Python-based Deep Learning models much more efficiently than RDD-based mapping.

#### Why Airflow?

We needed a way to manage dependencies. "Don't run the detailed analysis until we have at least 100 reviews". Airflow's DAG structure makes these dependencies explicit and easy to manage visually.

---

**(End of Part 2)**

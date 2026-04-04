# FINAL PROJECT REPORT

## REAL-TIME ASPECT-BASED SENTIMENT ANALYSIS FOR E-COMMERCE REVIEWS

**Course**: SE363 - Application Development on Big Data Platforms
**Student Team**: [Student Names]
**Instructor**: [Instructor Name]

---

# PART 1: INTRODUCTION & THEORETICAL BACKGROUND

## CHAPTER 1: INTRODUCTION

### 1.1. Context and Problem Statement

In the last decade, Vietnam has witnessed an unprecedented explosion in the E-commerce sector. Major platforms such as Lazada, Shopee, and Tiki have transformed from alternative shopping channels into primary hubs for consumer activity. According to recent reports from the Vietnam E-commerce Association (VECOM), the annual growth rate of the sector has consistently exceeded 20%, with millions of transactions processed daily.

With this growth comes a massive influx of user-generated content (UGC), specifically product reviews. A popular product, such as a budget smartphone or a trending fashion item, can easily accumulate tens of thousands of reviews. These reviews are a goldmine of information, offering authentic insights into product quality, seller performance, and delivery services.

However, the sheer volume of this data presents a "Big Data" challenge. For a human consumer, reading through thousands of reviews to make an informed purchase decision is impossible. For sellers and platform administrators, manually monitoring this feedback to maintain quality control is equally unfeasible.

Furthermore, traditional Sentiment Analysis approaches often fall short in this context. Most legacy systems treat sentiment analysis as a **binary classification problem** (Positive vs. Negative) or a **ternary classification problem** (Positive, Negative, Neutral) applied to the entire text. This "coarse-grained" approach fails to capture the nuance of real-world feedback.

Consider the following review:

> _"The phone looks amazing and the screen is sharp, but the battery drains very fast and the delivery took forever."_

A traditional system might label this review as "Neutral" (canceling out positive and negative) or incorrectly classify it based on the dominant keyword. In reality, this review contains specific sentiments for specific aspects:

- **Product Design**: Positive (_"amazing"_)
- **Screen Quality**: Positive (_"sharp"_)
- **Battery Life**: Negative (_"drains very fast"_)
- **Delivery Service**: Negative (_"took forever"_)

This limitation necessitates the development of an **Aspect-Based Sentiment Analysis (ABSA)** system capable of understanding these granular distinctions. Moreover, given the high velocity of e-commerce transactions, this analysis needs to happen in **Real-Time** to allow for immediate actionable insights.

### 1.2. Project Objectives

This project aims to build a comprehensive, end-to-end system that addresses the aforementioned challenges. Our primary objectives are fourfold:

#### 1. Constructing a Robust Big Data Pipeline

We aim to architect a scalable data processing pipeline capable of handling high-velocity data streams. By integrating **Apache Kafka** as a distributed message bus and **Apache Spark** for stream processing, we ensure the system can ingest, buffer, and process reviews in real-time without data loss or bottlenecks.

#### 2. Developing a Vietnamese-Specific Deep Learning Model

Generic NLP models often perform poorly on Vietnamese text due to its unique linguistic characteristics (such as compound words and tone marks). We aim to fine-tune **PhoBERT**, a pre-trained language model optimized for Vietnamese, to solve the **Multi-Polarity ABSA** problem. The model must distinguish sentiments across 9 distinct e-commerce aspects.

#### 3. Optimizing for Low-Latency Performance

A "Real-Time" system is defined by its responsiveness. Our technical objective is to minimize the end-to-end latency—measured from the moment a review is crawled to when its analysis is available on the dashboard—to under **5 seconds**. This involves optimizing model inference using Batch Processing and zero-copy data transfer techniques (Pandas UDF).

#### 4. Delivering Actionable Visualization

Raw data is useless without interpretation. We aim to build an interactive **Dashboard** that aggregates sentiment trends, highlights problem areas (e.g., a sudden spike in negative delivery reviews), and allows users to filter data by product, time, or aspect.

### 1.3. Scope of Research

- **Data Source**: The system targets product reviews from **Lazada.vn**, one of the top e-commerce platforms in Vietnam.
- **Target Aspects**: We define 9 core aspects relevant to the Vietnamese e-commerce domain:
  1.  Product Quality (Chất lượng sản phẩm)
  2.  Price (Giá cả)
  3.  Shipping/Delivery (Vận chuyển)
  4.  Packaging (Đóng gói)
  5.  Seller Service/Attitude (Thái độ phục vụ)
  6.  Authenticity (Tính xác thực)
  7.  Description Accuracy (Đúng mô tả)
  8.  Performance/Experience (Hiệu năng/Trải nghiệm)
  9.  Warranty/Return Policy (Bảo hành/Đổi trả)
- **Technology Stack**: The project is implemented using the Python ecosystem, heavily utilizing **PyTorch** for AI, **Apache Kafka** & **Spark** for Big Data, and **Docker** for containerization.

### 1.4. Practical Significance

The successful deployment of this system offers tangible benefits to all stakeholders in the e-commerce ecosystem:

- **For Sellers**: It acts as an early warning system. If a batch of products is defective or a shipping partner is underperforming, the "Sentiment Dashboard" will show a red alert in the relevant aspect column, allowing sellers to intervene before their reputation is ruined.
- **For Buyers**: It saves time. Instead of reading pages of text, a buyer can look at a summary: "Product Quality: 90% Positive", "Shipping: 40% Negative". This transparency aids in smarter purchasing decisions.
- **For Platforms**: It facilitates market intelligence. Platforms can aggregate this data to rank sellers, identify fraudulent behaviors (via Authenticity analysis), and understand broad market trends.

---

## CHAPTER 2: THEORETICAL BACKGROUND

### 2.1. Aspect-Based Sentiment Analysis (ABSA)

#### 2.1.1. Definition

Sentiment Analysis (SA), also known as Opinion Mining, is the field of study that analyzes people's opinions, sentiments, evaluations, attitudes, and emotions from written language. **Aspect-Based Sentiment Analysis (ABSA)** allows for a finer-grained analysis by identifying the sentiment expressed towards specific aspects or features of entities.

#### 2.1.2. Problem Formulation

In this project, we model ABSA as a **Multi-label Multi-class Classification** problem.
Given an input sentence $S = \{w_1, w_2, ..., w_n\}$, and a predefined set of aspects $A = \{a_1, a_2, ..., a_9\}$.
The goal is to determine, for each aspect $a_i \in A$, a set of sentiment polarities $P_i \subseteq \{Positive, Neutral, Negative\}$.

Note the set notation $P_i$. Unlike traditional ABSA which assigns a single label, our system supports **Multi-Polarity**. This means a single aspect in a single sentence can simultaneously effectively hold conflicting sentiments (e.g., "Good quality but bad finish" implies both Positive and Negative for 'Quality').

#### 2.1.3. Challenges in Vietnamese ABSA

- ** Implicit Aspects**: Users often don't name the aspect explicitly. E.g., _"Wait 2 weeks to receive"_ refers to _Shipping_, even though the word "shipping" is absent.
- **Negation & Slang**: Vietnamese reviews are full of slang (e.g., "kđ" for "không đùa", "gato" for jealous) and complex negation structures that simple bag-of-words models cannot capture.
- **Data Imbalance**: E-commerce data is heavily skewed towards positive ratings, making it difficult for models to learn negative patterns without specific sampling strategies.

### 2.2. The Transformer Architecture & BERT

The field of NLP was revolutionized in 2017 with the introduction of the **Transformer** architecture by Vaswani et al. Prior to this, Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTMs) were the state-of-the-art. However, these sequential models suffered from two main issues: inability to parallelize training (due to sequential dependency) and difficulty in retaining long-term dependencies in long sequences.

#### 2.2.1. Self-Attention Mechanism

The core innovation of the Transformer is the **Self-Attention** mechanism. It allows the model to weigh the importance of different words in a sentence relative to the word currently being processed.

Mathematically, for each word, we generate three vectors: Query ($Q$), Key ($K$), and Value ($V$). The attention score is calculated as:
$$ Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V $$
Here, the dot product $QK^T$ measures the similarity between the Query of the current word and the Keys of all other words. Dividing by $\sqrt{d_k}$ scales the dot products to prevent vanishing gradients in the softmax function. The result is a weighted sum of Value vectors, representing the context-aware embedding of the word.

#### 2.2.2. BERT (Bidirectional Encoder Representations from Transformers)

**BERT**, introduced by Google in 2018, utilizes a stack of Transformer Encoders. Unlike GPT (which uses Decoders and is uni-directional), BERT is **deeply bidirectional**. It is pre-trained on two tasks:

1.  **Masked Language Modeling (MLM)**: Randomly masking 15% of tokens and requiring the model to predict them based on context. This forces the model to understand the relationship between words in both directions.
2.  **Next Sentence Prediction (NSP)**: Predicting if sentence B logically follows sentence A.

This pre-training allows BERT to learn rich, contextual representations of language, which can then be **fine-tuned** on specific downstream tasks like ours (Classification) with minimal architecture changes.

### 2.3. PhoBERT: Adapting BERT for Vietnamese

While Multilingual BERT (mBERT) supports Vietnamese, it is suboptimal because it treats Vietnamese words as sequences of Latin sub-tokens without understanding the concept of "compound words" in Vietnamese.

**PhoBERT** (Nguyen & Nguyen, 2020) addresses this by being pre-trained specifically on a massive Vietnamese corpus (20GB of text). Key differences include:

- **Tokenizer**: PhoBERT uses a tokenizer combined with a Vietnamese word segmenter (e.g., recognizing "sản_phẩm" as one token, whereas mBERT might split it into "sản" and "phẩm").
- **Pre-training Data**: Trained on Vietnamese news, wiki, and social media text, allowing it to capture local nuances and idioms better than generic models.

In this project, `vinai/phobert-base` serves as the feature extractor backbone.

### 2.4. Big Data Technologies

#### 2.4.1. Apache Kafka

Apache Kafka is a distributed event streaming platform. In our architecture, it serves as the central nervous system.

- **Decoupling**: It separates the _Producers_ (Crawlers) from the _Consumers_ (Spark). If the crawler runs faster than the model can predict, Kafka buffers the data, preventing system crashes.
- **Durability**: Messages in Kafka are persisted to disk and replicated across the cluster, ensuring zero data loss even if a node fails.
- **Ordering**: Kafka guarantees order within a partition. We leverage this by partitioning data by `Product_ID`, ensuring reviews for the same product are grouped logically.

#### 2.4.2. Apache Spark & Structured Streaming

Apache Spark is a unified analytics engine for large-scale data processing.

- **Structured Streaming**: This API allows us to express streaming computations the same way as batch computations on static data. It treats the data stream as an unbounded Input Table. As new data arrives, rows are appended to this table.
- **Micro-batch Processing**: Spark processes data streams in small batches (e.g., every 1 second). This creates a "near real-time" effect while allowing for high-throughput optimizations (like vectorization) that are impossible with record-at-a-time processing.

#### 2.4.3. Apache Airflow

Apache Airflow is a platform to programmatically author, schedule, and monitor workflows.

- **DAGs (Directed Acyclic Graphs)**: We define our pipeline as code (Python). A DAG might look like: `Crawl -> Check_Data -> Train_Model -> Deploy`.
- **Extensibility**: Airflow allows us to unite the diverse technologies in our stack (Python scripts, Spark jobs, SQL queries) into a single, observable workflow.

---

**(End of Part 1)**

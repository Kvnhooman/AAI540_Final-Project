# AAI-540 Group 1 — Yelp Review Sentiment Classification

Predicts whether a Yelp review is positive (4-5 stars) or negative (1-2 stars) using engineered text features and a binary classifier.

---

## Project Structure

```
notebooks/
├── 01_setup_athena.ipynb        # sets up Glue database and Athena tables
├── 02_eda.ipynb                 # exploratory data analysis
├── 03_feature_engineering.ipynb # feature engineering + Feature Store
├── 04_split.ipynb               # train/test/validation/production split
├── 05_modeling.ipynb            # model training and evaluation
└── project_config.json          # auto-generated, shared config across notebooks
```

---

## Dataset

Yelp reviews split into two parquet files stored in a shared public S3 bucket:

| File | Rows | Period |
|------|------|--------|
| yelp_reviews_2019.parquet | 907,284 | 2019 |
| yelp_reviews_2020-2022.parquet | 1,204,411 | 2020-2022 |

S3 bucket: `s3://aai-540-group1-yelp-reviews/`

Schema:
```
review_id    string
user_id      string
business_id  string
stars        bigint   (1-5)
useful       bigint
funny        bigint
cool         bigint
text         string
date         timestamp
```

---

## Architecture

```
S3 Data Lake (public)
    ├── data/2019/                    raw 2019 parquet
    ├── data/2020_2022/               raw 2020-2022 parquet
    ├── features/yelp_features.parquet  engineered features
    ├── splits/train.parquet
    ├── splits/test.parquet
    ├── splits/validation.parquet
    ├── splits/production.parquet
    ├── athena-results/{your_name}/   per-person Athena query results
    └── feature-store/                SageMaker Feature Store offline store
         ↑
    AWS Glue (yelp_reviews_db)
         ↑
    Amazon Athena
         ↑
    SageMaker Notebooks
```

---

## What We Built

### 01 — Athena Setup
- Created Glue database `yelp_reviews_db`
- Manually defined Glue tables pointing to S3 parquet folders
- Verified Athena can query both tables

### 02 — EDA
- Loaded 907k reviews from Athena
- Analyzed star distribution — found 51% five-star reviews (class imbalance)
- Analyzed review length, engagement features, and time trends
- Zero nulls, duplicates, or empty reviews found

### 03 — Feature Engineering
- Sampled 100k rows randomly from 2019 data
- Dropped 3-star reviews (ambiguous) — ~8% of data
- Created binary target: 1-2 stars = negative (0), 4-5 stars = positive (1)
- Engineered features:
  - `review_length` — character count
  - `word_count` — word count
  - `vader_score` — VADER compound sentiment score (-1 to 1)
- Saved features to S3 and registered SageMaker Feature Store group

### 04 — Data Split
- Train/test/validation from 2019 data (stratified)
- Production data from 2020-2022 — simulates real deployment on newer reviews
- All splits saved to `s3://aai-540-group1-yelp-reviews/splits/`

| Split | Source | Size |
|-------|--------|------|
| Train | 2019 | ~40% |
| Test | 2019 | ~10% |
| Validation | 2019 | ~10% |
| Production | 2020-2022 | ~100k sampled |

---

## How to Reproduce

### Prerequisites

#### Regular AWS users (one-time setup)
1. Create a SageMaker domain in `us-east-2` (Ohio)
2. Go to IAM → Roles → your SageMaker execution role:
   - Permissions tab → attach `AWSGlueServiceRole`
   - Trust relationships tab → add `glue.amazonaws.com` to Service list:
     ```json
     {
       "Service": ["sagemaker.amazonaws.com", "glue.amazonaws.com"]
     }
     ```

#### AWS Academy users
No setup needed — `LabRole` already has all required permissions.

### Steps

1. Open SageMaker Studio in `us-east-2`
2. Upload all notebooks to your JupyterLab space
3. Run notebooks **in order**:

```
01_setup_athena.ipynb     ← run once, sets up Glue + Athena
02_eda.ipynb              ← optional, exploration only
03_feature_engineering.ipynb  ← run once, saves features to S3
04_split.ipynb            ← run once, saves splits to S3
05_modeling.ipynb         ← run anytime
```

4. In `01_setup_athena.ipynb` change `YOUR_NAME` to your initials:
```python
YOUR_NAME = 'studentA'  # change this
```

### Re-running After Session Expires (Academy)

AWS Academy sessions reset every ~4 hours. After a reset:
- Re-run `01_setup_athena.ipynb` from Cell 2 onwards — all create calls skip if resources exist
- Re-run `03_feature_engineering.ipynb` Cell 1 only (setup) to restore session
- Then continue with whichever notebook you need

---

## Features

| Feature | Type | Description |
|---------|------|-------------|
| `review_length` | int | Number of characters in review text |
| `word_count` | int | Number of words in review text |
| `useful` | int | Number of useful votes |
| `funny` | int | Number of funny votes |
| `cool` | int | Number of cool votes |
| `vader_score` | float | VADER compound sentiment score (-1 to 1) |
| `sentiment` | int | Target: 0 = negative, 1 = positive |

---

## Key Findings from EDA

- 907k reviews in 2019, zero data quality issues
- Star distribution heavily skewed toward 5 stars (51.1%)
- Average review length: 523 characters / 97 words
- 3-star reviews (8.2%) dropped as ambiguous for binary classification
- After dropping 3-stars: 73.4% positive, 26.6% negative
- Class imbalance handled via `class_weight='balanced'` in model

---

## Dependencies

All installed automatically in each notebook via `install_if_missing()`:
- `boto3` — AWS SDK
- `pandas` — data manipulation
- `nltk` — VADER sentiment analysis
- `pyarrow` — parquet support
- `scikit-learn` — model training and splitting

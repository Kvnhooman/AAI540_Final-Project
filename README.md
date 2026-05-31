# AAI-540 Group 1 — Yelp Review Sentiment Classification

Predicts whether a Yelp review is positive (4-5 stars) or negative (1-2 stars) using engineered text features and a binary classifier.

> **AI assistance:** This project used Claude (Anthropic) to help write code, debug AWS/SageMaker integration issues, and draft portions of this README. All design decisions, data analysis, and final code review were performed by Group 1 members.

---

## Results

Random Forest outperforms a simple VADER-threshold benchmark by ~4 percentage points F1 on held-out test data. Performance holds up on unseen 2020-2022 production data with no drift detected across any of the six engineered features, confirmed by Population Stability Index measurement against the training baseline.

Exact metrics vary slightly across scheduled pipeline runs due to 80% bootstrap subsampling; see `07_sagemaker_pipeline.ipynb` Section 15 for live execution history. The model is registered in the SageMaker Model Registry under model package group `yelp-sentiment-model-group`, with versioned lineage across runs. Prediction monitoring, drift metrics, and alarms are available in the CloudWatch dashboard `yelp-sentiment-monitoring`.

---

## Project Structure

```
notebooks/
├── 01_setup_athena.ipynb              # sets up Glue database and Athena tables
├── 02_eda.ipynb                       # exploratory data analysis
├── 03_feature_engineering.ipynb       # feature engineering + Feature Store
├── 04_split.ipynb                     # train/test/validation/production split
├── 05_modeling.ipynb                  # local model comparison (RF vs LogReg)
├── 06_sagemaker_model_store.ipynb     # SageMaker training, batch transform,
│                                      # Model Registry, Model Card
├── 07_sagemaker_pipeline.ipynb        # CI/CD pipeline DAG + daily EventBridge
│                                      # schedule with bootstrap subsampling
├── 08_monitoring.ipynb                # CloudWatch prediction logging, PSI drift
│                                      # check, dashboard, and alarms
└── project_config.json                # auto-generated, shared config across notebooks
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
    ├── data/2019/                      raw 2019 parquet
    ├── data/2020_2022/                 raw 2020-2022 parquet
    ├── features/yelp_features.parquet  engineered features
    ├── splits/train.parquet
    ├── splits/test.parquet
    ├── splits/validation.parquet
    ├── splits/production.parquet
    ├── athena-results/{your_name}/     per-person Athena query results
    └── feature-store/                  SageMaker Feature Store offline store
         ↑
    AWS Glue (yelp_reviews_db)
         ↑
    Amazon Athena
         ↑
    SageMaker Notebooks
         ↓ (notebook 08 scores production and publishes monitoring data)
    Amazon CloudWatch
        ├── Logs  /yelp-sentiment/predictions    structured prediction events
        ├── Metrics  YelpSentiment/Monitoring     volume, positive rate, confidence, drift PSI
        └── Dashboard + Alarms                   yelp-sentiment-monitoring
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
- Train/test/validation from 2019 data (stratified, `random_state=42`)
- Production data from 2020-2022 — simulates real deployment on newer reviews

| Split | Source | Rows | % of cleaned 2019 |
|-------|--------|------|-------------------|
| Train | 2019 | ~61k | ~67% |
| Validation | 2019 | ~15k | ~16.65% |
| Test | 2019 | ~15k | ~16.65% |
| Production | 2020-2022 | ~93k | — |

Two-stage split via `train_test_split`: first 2019 data into train/temp (66.7/33.3), then temp into validation/test (50/50). Production set is sampled from 2020-2022 data and held entirely out of training.

### 05 — Modeling

Trained Logistic Regression as a local baseline model and Random Forest as the improved model.
Both models used `class_weight='balanced'` to handle the 73/27 class imbalance.
Random Forest performed best, achieving:

- Test F1: 0.9365
- Test ROC-AUC: 0.9517
- Production F1: 0.9372
- Production ROC-AUC: 0.9555

Model performance remained consistent on 2020–2022 production data.

### 06 — SageMaker Model Store

Trained the Random Forest as a managed SageMaker SKLearn training job using `ml.m5.large`.
Compared the SageMaker-trained model against a simple VADER-threshold benchmark:

- Benchmark test F1: 0.9188
- SageMaker Random Forest test F1: 0.9365
- Benchmark test ROC-AUC: 0.8997
- SageMaker Random Forest test ROC-AUC: 0.9517

Deployed the model using SageMaker Batch Transform on a production sample, registered the model in SageMaker Model Registry, and created a SageMaker Model Card.

### 07 — SageMaker Pipeline (CI/CD DAG)
- Built an end-to-end SageMaker Pipeline orchestrating Train → Evaluate → Condition → (Register | Fail)
- Conditional registration gated on `F1 ≥ MinF1Threshold` parameter
- Demonstrated both success (registers model) and failure (`FailStep` terminates pipeline) terminal states
- Scheduled the pipeline via EventBridge Scheduler to run every 24 hours with `SampleSeed=-1` and `SampleFrac=0.8` — each daily run trains on a different 80% random subset, producing meaningfully varied metrics
- Each scheduled execution adds a new version to the Model Package Group, giving the registry real version history over time

### 08 — Monitoring
- Rebuilt the production model and scored the full 2020-2022 production set (93k rows)
- Logged 1,000 sampled predictions to CloudWatch Logs (`/yelp-sentiment/predictions`) as structured JSON events, each carrying the review id, predicted label, model score, confidence, and true label
- Published summary metrics to custom namespace `YelpSentiment/Monitoring`: prediction volume, positive prediction rate, mean confidence
- Ran a Population Stability Index (PSI) drift check comparing the distribution of each feature in production against the training baseline — confirmed no significant drift on any feature
- Published `MaxFeaturePSI` as a CloudWatch metric so drift is a monitorable, alertable number rather than a static claim
- Created the CloudWatch dashboard `yelp-sentiment-monitoring` with metric time-series charts and a live Logs Insights prediction table
- Created two CloudWatch alarms: `yelp-sentiment-low-confidence` (MeanConfidence below 0.65) and `yelp-sentiment-feature-drift` (MaxFeaturePSI above 0.25)
- Saved all monitoring resource names back to `project_config.json`

---

## How to Reproduce

### Prerequisites

#### Regular AWS users (one-time setup)
1. Create a SageMaker domain in `us-east-2` (Ohio) — this matches the shared bucket
2. Go to IAM → Roles → your SageMaker execution role:
   - Permissions tab → attach `AWSGlueServiceRole`
   - Trust relationships tab → add `glue.amazonaws.com` to Service list:
     ```json
     {
       "Service": ["sagemaker.amazonaws.com", "glue.amazonaws.com"]
     }
     ```

#### AWS Academy users
`LabRole` already has the required IAM permissions, but Academy typically locks you into `us-east-1` while the shared dataset bucket lives in `us-east-2`. Glue and Athena are region-scoped, so cross-region access is not reliable — you need a copy of the dataset in your Academy region.

1. Confirm your Academy region (top-right of the AWS console). If it's already `us-east-2`, you can use the shared bucket directly and skip step 2.
2. If your region is different (most commonly `us-east-1`):
   - Create your own S3 bucket in that region, e.g. `s3://yourname-yelp-data/`
   - Copy the two Yelp parquet files into it under the same prefixes the notebooks expect:
     ```
     s3://yourname-yelp-data/data/2019/yelp_reviews_2019.parquet
     s3://yourname-yelp-data/data/2020_2022/yelp_reviews_2020-2022.parquet
     ```
     Easiest way (run from any machine with AWS credentials for both accounts, or from a SageMaker terminal):
     ```bash
     aws s3 cp s3://aai-540-group1-yelp-reviews/data/2019/yelp_reviews_2019.parquet \
               s3://yourname-yelp-data/data/2019/ \
               --source-region us-east-2 --region us-east-1
     aws s3 cp s3://aai-540-group1-yelp-reviews/data/2020_2022/yelp_reviews_2020-2022.parquet \
               s3://yourname-yelp-data/data/2020_2022/ \
               --source-region us-east-2 --region us-east-1
     ```
   - In `01_setup_athena.ipynb` Section 1, update `REGION` and `SOURCE_BUCKET` to your values. All downstream notebooks pick these up automatically via `project_config.json`.

### Steps

1. Open SageMaker Studio in `us-east-2`
2. Upload all notebooks to your JupyterLab space
3. Run notebooks in order:

```
01_setup_athena.ipynb            run once, sets up Glue + Athena
02_eda.ipynb                     optional, exploration only
03_feature_engineering.ipynb     run once, saves features to S3
04_split.ipynb                   run once, saves splits to S3
05_modeling.ipynb                run anytime (local model comparison)
06_sagemaker_model_store.ipynb   trains in SageMaker, registers model + card
07_sagemaker_pipeline.ipynb      builds CI/CD pipeline DAG; Section 14 creates
                                 a daily EventBridge schedule (remember to
                                 stop it via Section 16 when done)
08_monitoring.ipynb              run after all other notebooks; scores the
                                 production split, logs predictions to CloudWatch,
                                 checks feature drift (PSI), and creates a
                                 dashboard and alarms
```

4. In `01_setup_athena.ipynb` change `YOUR_NAME` to your initials:
```python
YOUR_NAME = 'Your_name/StudentID Here'  # change this
```

### Coming Back After a Session Break

Academy lab sessions have time limits (typically a few hours), but the AWS resources you've created — S3 buckets, Glue databases, SageMaker model packages, pipelines — persist across sessions. What gets lost is your Studio kernel state (Python variables in memory).

When you come back:
- Reopen SageMaker Studio in the same region you set up in
- Open the notebook you were working in and run the Setup cells to restore Python state
- All the `create_*` calls in the notebooks are idempotent (use try/except on AlreadyExistsException), so re-running earlier cells won't error or duplicate resources

### Stopping the Daily Pipeline Schedule

Notebook 07 Section 14 creates an EventBridge schedule that fires the pipeline every 24 hours. To stop the recurring runs (e.g., when you're done collecting metric-variation data), run **notebook 07 Section 16** — it deletes the schedule. Existing execution history stays in SageMaker Pipelines; only the recurring trigger is removed.

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

## Key Findings

- 907k reviews in 2019, zero data quality issues
- Star distribution heavily skewed toward 5 stars (51.1%)
- 3-star reviews (8.2%) dropped as ambiguous for binary classification
- After dropping 3-stars: 73.4% positive, 26.6% negative
- Random Forest (90.9% accuracy, 0.954 ROC-AUC) outperformed Logistic Regression
- Model performance consistent across validation, test, and 2020-2022 production data — no drift detected

---

## Dependencies

All installed automatically in each notebook via `install_if_missing()`:
- `boto3` — AWS SDK
- `pandas` — data manipulation
- `nltk` — VADER sentiment analysis
- `pyarrow` — parquet support
- `scikit-learn` — model training and splitting
- `sagemaker` (>=2,<3) — training jobs, batch transform, Model Registry, Pipelines (notebooks 06, 07)

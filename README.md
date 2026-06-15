# AAI-540 Group 1 — Yelp Review Sentiment Classification

Predicts whether a Yelp review is positive (4-5 stars) or negative (1-2 stars) using engineered text features and a binary classifier, deployed and monitored as an end-to-end MLOps system on AWS SageMaker.

> **AI assistance:** This project used Claude (Anthropic) to help write code, debug AWS/SageMaker integration issues, and draft portions of this README. All design decisions, data analysis, and final code review were performed by Group 1 members.

---

## Results

Random Forest outperforms a simple VADER-threshold benchmark by ~2 percentage points F1 on held-out test data (test F1 0.936 vs 0.919). Performance holds up on unseen 2020-2022 production data with no drift detected across any of the six engineered features, confirmed by Population Stability Index (PSI) measurement against the training baseline (max feature PSI ≈ 0.003, well below the 0.10 "no shift" threshold).

The model is registered in the SageMaker Model Registry under model package group `yelp-sentiment-model-group`, with versioned lineage across runs. Exact metrics vary slightly across scheduled pipeline runs due to 80% bootstrap subsampling; see `07_sagemaker_pipeline.ipynb` Section 15 for live execution history. Prediction monitoring, drift metrics, and alarms are available in the CloudWatch dashboard `yelp-sentiment-monitoring`. A SageMaker Processing bias monitor checks fairness across a review-length proxy facet and flags a demographic-parity gap, demonstrating that the monitoring layer actively detects and reports bias rather than silently passing.

---

## Project Structure

```
01_setup_athena.ipynb              # sets up Glue database and Athena tables
02_eda.ipynb                       # exploratory data analysis
03_feature_engineering.ipynb       # feature engineering + Feature Store (create + ingest)
04_split.ipynb                     # train/test/validation/production split
05_modeling.ipynb                  # local model comparison (RF vs LogReg)
06_sagemaker_model_store.ipynb     # SageMaker training, batch transform,
                                   # Model Registry, Model Card
07_sagemaker_pipeline.ipynb        # CI/CD pipeline DAG + daily EventBridge
                                   # schedule with bootstrap subsampling
08_monitoring.ipynb                # CloudWatch prediction logging, PSI drift
                                   # check, dashboard, and alarms
09_bias_monitoring.ipynb           # model bias monitor via SageMaker Processing Job
10_monitoring_reports.ipynb        # model/data/infra/bias monitoring reports
                                   # (embeds live CloudWatch + bias values)
yelp_json_to_parquet.ipynb         # one-time raw JSON -> Parquet conversion + S3 upload
project_config.json                # auto-generated, shared config across notebooks
```

---

## Dataset

Yelp reviews split into two parquet files stored in a shared S3 bucket:

| File | Rows | Period |
|------|------|--------|
| yelp_reviews_2019.parquet | 907,284 | 2019 |
| yelp_reviews_2020-2022.parquet | 1,204,411 | 2020-2022 |

S3 bucket: `s3://aai540-group1-yelp-data/` (region `us-east-1`)

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
S3 Data Lake  (s3://aai540-group1-yelp-data/, us-east-1)
    ├── data/2019/                      raw 2019 parquet
    ├── data/2020_2022/                 raw 2020-2022 parquet
    ├── features/yelp_features.parquet  engineered features
    ├── splits/{train,validation,test,production}.parquet
    ├── athena-results/{your_name}/     per-person Athena query results
    ├── feature-store/                  SageMaker Feature Store offline store
    ├── sentiment-model-store/          training inputs, model artifacts, batch output
    ├── bias-monitor/                   bias Processing Job inputs + reports
    └── monitoring-reports/             model/data/infra/bias report artifacts
         ↑
    AWS Glue (yelp_reviews_db)  ←→  Amazon Athena
         ↑
    SageMaker Notebooks
         ├── Training Job (RandomForest) ──► Model Registry (yelp-sentiment-model-group)
         ├── Batch Transform ─────────────► scored predictions in S3
         ├── SageMaker Pipeline (CI/CD DAG) ─► EventBridge daily schedule
         └── Processing Job (bias monitor) ─► bias summary.json in S3
         ↓ (notebook 08 scores production and publishes monitoring data)
    Amazon CloudWatch
        ├── Logs  /yelp-sentiment/predictions    structured prediction events
        ├── Metrics  YelpSentiment/Monitoring     volume, positive rate, confidence, drift PSI
        └── Dashboard + Alarms                   yelp-sentiment-monitoring
```

---

## What We Built

### 00 — Data Upload to S3 (`yelp_json_to_parquet.ipynb`)

- Downloaded the public Yelp reviews dataset
- Converted raw JSON review data to Parquet format for efficient analytics
- Partitioned data into 2019 reviews and 2020-2022 reviews
- Uploaded Parquet files to the Amazon S3 data lake and verified the object structure
- Established the project data source used by Athena, Feature Store, modeling, deployment, and monitoring notebooks

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
- Saved features to S3 and registered the SageMaker Feature Store group `yelp-review-features` (online + offline stores)
- Ingested feature records into the feature group and verified retrieval from the online store, so the store holds queryable records rather than just a schema

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

### 05 — Modeling and Evaluation
- Loaded train, validation, test, and production datasets from S3
- Built a benchmark **Logistic Regression** model using class balancing
- Built a **Random Forest Classifier** as the primary production model
- Evaluated both on accuracy, precision, recall, F1, and ROC-AUC across validation, test, and production
- Generated model evaluation and feature-importance visualizations
- Random Forest consistently outperformed Logistic Regression and was selected as the production model
- Saved model artifacts and evaluation outputs for deployment in subsequent notebooks

### 06 — SageMaker Model Store
Trained the Random Forest as a managed SageMaker SKLearn training job using `ml.m5.large`. Compared the SageMaker-trained model against a simple VADER-threshold benchmark:

- Benchmark test F1: 0.919
- SageMaker Random Forest test F1: 0.936
- Benchmark test ROC-AUC: 0.907
- SageMaker Random Forest test ROC-AUC: 0.954

Deployed the model using SageMaker Batch Transform on a production sample (predictions written to S3), registered the model in the SageMaker Model Registry, and created a SageMaker Model Card documenting intended use, training data, evaluation, and ethical considerations.

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

### 09 — Bias Monitoring (SageMaker Processing Job)
- Ran a model bias monitor as a managed **SageMaker Processing Job**, comparing the 2019 training baseline against the 2020-2022 production window
- Used `review_length_group` (short vs long, split at the baseline median) as a **proxy facet** — the dataset has no demographic attributes, so in production this would be replaced with a business-approved protected attribute
- Computed disparate impact ratio, demographic parity difference, and false-negative-rate difference across groups, applied thresholds, and wrote report artifacts (including `summary.json`) back to S3
- Flagged a demographic-parity difference of ~0.16 (above the 0.10 threshold), showing the monitor actively detects and reports bias

### 10 — Monitoring Reports
- Generated formal report artifacts documenting the monitoring framework, each **embedding live values** pulled back from CloudWatch and S3 rather than static descriptions:
  - **Model Monitoring Report** — latest prediction volume, positive rate, mean confidence
  - **Data Monitoring Report** — latest MaxFeaturePSI and drift verdict
  - **Infrastructure Monitoring Report** — current CloudWatch alarm states + metric values
  - **Bias Monitoring Report** — latest verdict read from notebook 09's `summary.json`
- Uploaded all four reports to `s3://aai540-group1-yelp-data/monitoring-reports/<timestamp>/` for long-term storage and project documentation
- Provided the consolidated monitoring and reporting layer for the Yelp Sentiment MLOps workflow

---

## How to Reproduce

This project runs on **AWS Academy (`LabRole`) in `us-east-1`** against the shared bucket `s3://aai540-group1-yelp-data/`. Glue and Athena are region-scoped, so the dataset and all resources must live in the same region you run in.

### Prerequisites

#### AWS Academy users (primary path)
`LabRole` already has the required IAM permissions (SageMaker, S3, Glue, Athena, CloudWatch). Confirm your Academy region is `us-east-1` (top-right of the AWS console) and that the dataset is present at:
```
s3://aai540-group1-yelp-data/data/2019/yelp_reviews_2019.parquet
s3://aai540-group1-yelp-data/data/2020_2022/yelp_reviews_2020-2022.parquet
```
If you are running in a different region or your own account, create a bucket in your region, copy the two parquet files into it under the same prefixes, and update `REGION` and `SOURCE_BUCKET` in `01_setup_athena.ipynb` Section 1. All downstream notebooks pick these up automatically via `project_config.json`.

#### Regular (non-Academy) AWS users
1. Create a SageMaker domain in the same region as your data bucket
2. Go to IAM → Roles → your SageMaker execution role:
   - Permissions tab → attach `AWSGlueServiceRole`
   - Trust relationships tab → add `glue.amazonaws.com` to the Service list:
     ```json
     {
       "Service": ["sagemaker.amazonaws.com", "glue.amazonaws.com"]
     }
     ```

### Steps

1. Open SageMaker Studio in `us-east-1`
2. Upload all notebooks to your JupyterLab space
3. Run notebooks in order:

```
01_setup_athena.ipynb            run once, sets up Glue + Athena
02_eda.ipynb                     optional, exploration only
03_feature_engineering.ipynb     run once, saves features to S3 + ingests Feature Store records
04_split.ipynb                   run once, saves splits to S3
05_modeling.ipynb                run anytime (local model comparison)
06_sagemaker_model_store.ipynb   trains in SageMaker, registers model + card, batch transform
07_sagemaker_pipeline.ipynb      builds CI/CD pipeline DAG; Section 14 creates a daily
                                 EventBridge schedule (remember to stop it via Section 16)
08_monitoring.ipynb              scores the production split, logs predictions to CloudWatch,
                                 checks feature drift (PSI), and creates a dashboard + alarms
09_bias_monitoring.ipynb         runs the bias monitor as a SageMaker Processing Job
10_monitoring_reports.ipynb      run last (after 08 + 09); generates monitoring reports
                                 with live CloudWatch + bias values
```

4. In `01_setup_athena.ipynb` change `YOUR_NAME` to your initials:
```python
YOUR_NAME = 'Your_name/StudentID Here'  # change this
```

### Coming Back After a Session Break

Academy lab sessions have time limits (typically a few hours), but the AWS resources you've created — S3 buckets, Glue databases, SageMaker model packages, pipelines, feature groups — persist across sessions. What gets lost is your Studio kernel state (Python variables in memory).

When you come back:
- Reopen SageMaker Studio in `us-east-1`
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
- Random Forest (90.9% accuracy, 0.954 ROC-AUC) outperformed Logistic Regression and the VADER benchmark
- Model performance consistent across validation, test, and 2020-2022 production data — no feature drift detected (max PSI ≈ 0.003)
- Bias monitor flagged a demographic-parity gap (~0.16) across the review-length proxy facet, confirming the fairness monitoring works end to end

---

## Dependencies

All installed automatically in each notebook via `install_if_missing()`:
- `boto3` — AWS SDK
- `pandas` — data manipulation
- `nltk` — VADER sentiment analysis
- `pyarrow` — parquet support
- `scikit-learn` — model training and splitting
- `sagemaker` (>=2,<3) — training jobs, batch transform, Model Registry, Pipelines (notebooks 06, 07)

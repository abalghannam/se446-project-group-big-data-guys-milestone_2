# SE446 Milestone 2 — Big Data Guys

Spark DataFrame analytics + MLlib arrest predictor on Chicago Crime data
(Hadoop 3.4.1 / Spark 3.5.4, 1 master + 2 workers).

## Team

| Member               | ID     | GitHub          | Tasks |
|----------------------|--------|-----------------|-------|
| Abdulrahman Alghannam | 220455 | `abalghannam`  | 1, 11 |
| Khalid Aleisa         | 230525 | `khalidaleissa`| 5, 6  |
| Abdulmohsen Binkhamis | 230241 | `amohsentk`    | 2, 7  |
| Saud Aldawood         | 230336 | `saudaldawood` | 3, 10 |
| Feras Alkahtani       | 230313 | `Feras1972-KHT`| 4, 9  |

## Spec compliance (May 2026 update)

1. **Task 8 (CrossValidator) is omitted** — waived by the instructor.
2. **Phase B (Tasks 5–7) trains on a 5% sample** via `df.sample(fraction=0.05, seed=42)`.
   On the cluster this gives 39,534 rows (Train 31,728 / Test 7,806).
3. **Task 11 uses `--deploy-mode cluster`** with the application stdout pulled into
   `output/spark_submit/run.log`.

## Repository layout

```
.
├── M2_BigDataGuys.ipynb          Notebook (Tasks 1–7), executed locally
├── chicago_arrest_pipeline.py    Standalone Phase B script for spark-submit
├── scripts/
│   └── notebook_builder.py       Notebook generator
├── output/
│   ├── year_distribution.png     Task 3 chart
│   ├── cluster_yarn_log.txt      Task 10 evidence
│   └── spark_submit/
│       ├── console.log           Task 11 spark-submit invocation log
│       └── run.log               Task 11 application stdout
└── README.md
```

## Executive summary

We reproduce the four M1 MapReduce analyses with Spark DataFrames + Spark SQL on the
full 793,072-row HDFS dataset (numbers match M1 exactly). For arrest prediction we
build a Spark MLlib pipeline (StringIndexer × 2 + VectorAssembler + classifier) and
train Logistic Regression, Random Forest, and Gradient-Boosted Trees on a 5% sample
as required by the May 2026 spec update.

**Top model by AUC: GBT (0.8241).** Random Forest is a close second (0.8067) and is
substantially cheaper to train.

---

# Phase A — DataFrame analytics on the full dataset

## Task 1 — Crime type distribution
*Author: Abdulrahman Alghannam (220455, `abalghannam`)*

```python
crime_type_counts = (crimes_df
                     .groupBy("Primary Type")
                     .agg(fn.count(fn.lit(1)).alias("incident_count"))
                     .orderBy(fn.col("incident_count").desc()))
```

**M1 (MapReduce) ↔ M2 (Spark) — Top 10:**

| Crime type | M1 | M2 |
|------------|---:|---:|
| THEFT | 162,688 | 162,688 |
| BATTERY | 151,930 | 151,930 |
| CRIMINAL DAMAGE | 91,241 | 91,241 |
| NARCOTICS | 74,127 | 74,127 |
| ASSAULT | 54,070 | 54,070 |
| MOTOR VEHICLE THEFT | 48,494 | 48,494 |
| BURGLARY | 39,872 | 39,872 |
| OTHER OFFENSE | 36,893 | 36,893 |
| ROBBERY | 30,991 | 30,991 |
| DECEPTIVE PRACTICE | 30,396 | 30,396 |

Numbers match exactly. Spark's DataFrame engine keeps the aggregation in memory
rather than the disk shuffle that streaming MapReduce performs.

---

# Phase C — Deployment evidence

---

## Task 11 — spark-submit (cluster mode)
*Author: Abdulrahman Alghannam (220455, `abalghannam`)*

Per the May 2026 spec update, Task 11 uses `--deploy-mode cluster`:

```bash
abalghannam@master-node:~$ spark-submit --master yarn --deploy-mode cluster \
    --num-executors 2 --executor-memory 1g --executor-cores 1 \
    --driver-memory 1g chicago_arrest_pipeline.py
```

YARN application: `application_1777830883738_0023` — `final status: SUCCEEDED`.

Application stdout is pulled with `yarn logs -applicationId application_1777830883738_0023`
and saved to `output/spark_submit/run.log`. The console.log
(`output/spark_submit/console.log`) captures the spark-submit invocation and YARN's
progress reports.

Excerpt from `run.log`:

```
Spark version:  3.5.4
Master:         yarn
Full dataset rows: 793,072
Phase B sample: 39,534 rows  (5%, seed=42)
Train rows: 31,728  | Test rows: 7,806

>>> LogisticRegression
  AUC       0.6022
  Accuracy  0.7280
  F1        0.6376
  Train(s)  40.2

>>> RandomForest
  AUC       0.8067
  Accuracy  0.8156
  F1        0.7802
  Train(s)  53.0

>>> GBT
  AUC       0.8241
  Accuracy  0.8500
  F1        0.8337
  Train(s)  471.1

Top model by AUC: GBT (0.8241)
```

---

## Spec note — executor cores

The M2 spec lists `--executor-cores 2`. The course YARN cluster's maximum container
allocation is `<memory:1536, vCores:1>` — requesting 2 vcores returns
`InvalidResourceRequestException`. We therefore use `--executor-cores 1`, the same
setting M1 used.

---

## Member contributions

| Member | Tasks | Contribution |
|--------|-------|--------------|
| Abdulrahman Alghannam (`abalghannam`) | 1, 11 | Crime-type DataFrame query; spark-submit cluster-mode submission and log retrieval |
| Khalid Aleisa (`khalidaleissa`)        | 5, 6  | StringIndexer + VectorAssembler pipeline; three-classifier training and evaluation |
| Abdulmohsen Binkhamis (`amohsentk`)    | 2, 7  | Spark SQL location-hotspots query; Random Forest feature importances |
| Saud Aldawood (`saudaldawood`)         | 3, 10 | Year-trend table + matplotlib chart; yarn-client cluster execution evidence |
| Feras Alkahtani (`Feras1972-KHT`)      | 4, 9  | Arrest-rate analysis; local notebook execution evidence |

## How to reproduce

Locally:
```bash
python3 -m venv venv && source venv/bin/activate
pip install pyspark==3.5.1 pandas matplotlib jupyter numpy
jupyter nbconvert --to notebook --execute M2_BigDataGuys.ipynb --output M2_BigDataGuys.ipynb
```

On the cluster:
```bash
ssh <user>@134.209.172.50
source /etc/profile.d/hadoop.sh
source /etc/profile.d/spark.sh
# one-time deps for python3.12
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.12 get-pip.py --user
python3.12 -m pip install --user numpy 'setuptools>=68'
# Phase B standalone (cluster mode):
spark-submit --master yarn --deploy-mode cluster \
    --num-executors 2 --executor-memory 1g --executor-cores 1 \
    --driver-memory 1g chicago_arrest_pipeline.py
```

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

## Task 4 — Arrest rate analysis
*Author: Feras Alkahtani (230313, `Feras1972-KHT`)*

**Cluster — overall:** **221,932 / 793,073 = 27.98%** (matches M1 within rounding).

**Top arrest rates by crime type (min 100 records):**

| Crime type | Records | Arrest rate |
|------------|--------:|------------:|
| NARCOTICS | 74,127 | 99.88% |
| PROSTITUTION | 9,100 | 99.88% |
| LIQUOR LAW VIOLATION | 2,349 | 99.83% |
| GAMBLING | 1,314 | 99.77% |
| INTERFERENCE WITH PUBLIC OFFICER | 803 | 80.70% |
| WEAPONS VIOLATION | 8,893 | 74.60% |
| CRIMINAL TRESPASS | 21,476 | 73.58% |
| PUBLIC PEACE VIOLATION | 1,827 | 66.83% |
| HOMICIDE | 13,173 | 48.11% |
| SEX OFFENSE | 3,932 | 32.38% |

The arrest rate splits into two regimes — proactive-policing crimes near 100%
(the report only exists because an officer made the stop) and reactive-reporting
crimes like THEFT (14.2%) where most cases go unsolved. Phase B's ML model exploits
this structure.

---

# Phase B — MLlib arrest predictor (5% sample)

---

## Task 5 — Feature pipeline
*Author: Khalid Aleisa (230525, `khalidaleissa`)*

`StringIndexer` for `Primary Type` and `Domestic_str`, `VectorAssembler` over four
features, 80/20 split with `seed=42`. The 5% sample is applied before any feature
engineering.

Sample feature vectors from the cluster training set:

```
+-------------------+---------+----+--------+------------+-------+--------------------+-----+
|Primary Type       |crime_idx|Hour|District|Domestic_str|dom_idx|feat_vec            |label|
+-------------------+---------+----+--------+------------+-------+--------------------+-----+
|HOMICIDE           |11.0     |10  |25      |false       |0.0    |[10.0,11.0,25.0,0.0]|1    |
|HOMICIDE           |11.0     |13  |5       |false       |0.0    |[13.0,11.0,5.0,0.0] |1    |
|HOMICIDE           |11.0     |20  |3       |false       |0.0    |[20.0,11.0,3.0,0.0] |0    |
+-------------------+---------+----+--------+------------+-------+--------------------+-----+
```

Vector layout: `[Hour, crime_idx, District, dom_idx]`.

---

## Task 6 — Train and evaluate three classifiers
*Author: Khalid Aleisa (230525, `khalidaleissa`)*

Cluster results (5% sample of the full HDFS dataset):

| Model | Params | Train (s) | AUC | Accuracy | F1 | Precision | Recall |
|-------|--------|----------:|----:|---------:|---:|----------:|-------:|
| Logistic Regression | maxIter=100, regParam=0.01 | 21.3 | 0.6022 | 0.7280 | 0.6376 | 0.6923 | 0.7280 |
| Random Forest | numTrees=100, maxDepth=5, maxBins=64 | 36.1 | 0.8067 | 0.8156 | 0.7802 | 0.8528 | 0.8156 |
| **GBT** | maxIter=50, maxDepth=5, maxBins=64 | 437.7 | **0.8241** | **0.8500** | **0.8337** | **0.8610** | **0.8500** |

**Confusion matrices (TN/FP/FN/TP):**
- LR:  (5549, 93, 2030, 133)
- RF:  (5641, 1, 1438, 725)
- GBT: (5553, 89, 1082, 1081)

**Top model by AUC: GBT (0.8241).** GBT trains 12× longer than RF for ~2 percentage
points of AUC — for production deployment Random Forest is the better cost/quality
trade-off.

---

# Phase C — Deployment evidence

---

## Task 9 — Local execution
*Author: Feras Alkahtani (230313, `Feras1972-KHT`)*

Notebook executed end-to-end with `jupyter nbconvert --execute` (Python 3.9, PySpark
3.5.1, Java 17). Cell 1 prints:

```
Runtime:         local
Spark version:   3.5.1
Spark master:    local[*]
```

10,000 rows generated in-memory by the W09B-style synthetic generator. All Tasks 1–7
ran; outputs are embedded in `M2_BigDataGuys.ipynb`.

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

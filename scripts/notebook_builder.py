"""Build M2_BigDataGuys.ipynb from cell sources.

Run: python scripts/notebook_builder.py
"""
import json
from pathlib import Path

OUTPUT = "M2_BigDataGuys.ipynb"

# Author display strings (used by the {AUTHOR} substitutions in cell sources).
ABDULRAHMAN = "Abdulrahman Alghannam (220455)"
KHALID      = "Khalid Aleisa (230525)"
ABDULMOHSEN = "Abdulmohsen Binkhamis (230241)"
SAUD        = "Saud Aldawood (230336)"
FERAS       = "Feras Alkahtani (230313)"


_cells = []


def md(text):
    _cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })


def code(text):
    _cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    })


# ---------- Header ----------
md(f"""# SE446 Milestone 2 — Big Data Guys

Spark DataFrame analytics + MLlib arrest predictor on Chicago Crime data.

| Member               | ID     | GitHub          | Tasks |
|----------------------|--------|-----------------|-------|
| {ABDULRAHMAN}        | 220455 | abalghannam     | 1, 11 |
| {KHALID}             | 230525 | khalidaleissa   | 5, 6  |
| {ABDULMOHSEN}        | 230241 | amohsentk       | 2, 7  |
| {SAUD}               | 230336 | saudaldawood    | 3, 10 |
| {FERAS}              | 230313 | Feras1972-KHT   | 4, 9  |

**Spec compliance (May 2026 update):**
1. Task 8 (`CrossValidator`) is **omitted** — waived by the instructor.
2. Phase B (Tasks 5–7) trains on a **5% sample**: `df.sample(fraction=0.05, seed=42)`.
3. Task 11 uses `--deploy-mode cluster` and the application stdout is fetched with
   `yarn logs -applicationId <appId>` into `output/spark_submit/run.log`.
""")


# ---------- Setup ----------
md("---\n## 0. Spark session bootstrap")

code("""import os
import time
import shutil

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as fn
from pyspark.sql.types import IntegerType, StringType


def running_on_cluster() -> bool:
    return shutil.which("hdfs") is not None


_app_name = "M2_BigDataGuys"

if running_on_cluster():
    spark = (SparkSession.builder
             .appName(_app_name)
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    runtime = "cluster"
else:
    spark = (SparkSession.builder
             .appName(_app_name + "_local")
             .master("local[*]")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.driver.memory", "2g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    runtime = "local"

print("Runtime:        ", runtime)
print("Spark version:  ", spark.version)
print("Spark master:   ", spark.sparkContext.master)
""")


# ---------- Data load ----------
md("---\n## 1. Load the Chicago Crime dataset")
md("Cluster reads the real CSV from HDFS. Local generates 10K rows in memory.")

code("""HDFS_CSV = "hdfs:///data/chicago_crimes.csv"


def _load_real_data() -> "DataFrame":
    raw = spark.read.csv(HDFS_CSV, header=True, inferSchema=True)
    enriched = (raw
                .withColumn("Hour",
                            fn.hour(fn.to_timestamp(fn.col("Date"),
                                                    "MM/dd/yyyy hh:mm:ss a")))
                .withColumn("label",        fn.col("Arrest").cast(IntegerType()))
                .withColumn("Domestic_str", fn.col("Domestic").cast(StringType())))
    return enriched


def _load_synthetic_data(rows: int = 10_000) -> "DataFrame":
    import random
    random.seed(42)
    arrest_p_by_type = {
        "NARCOTICS":            0.85,
        "PROSTITUTION":         0.80,
        "WEAPONS VIOLATION":    0.60,
        "BATTERY":              0.30,
        "ASSAULT":              0.25,
        "ROBBERY":              0.15,
        "THEFT":                0.10,
        "BURGLARY":             0.08,
        "MOTOR VEHICLE THEFT":  0.06,
        "CRIMINAL DAMAGE":      0.05,
    }
    locs = ["STREET", "RESIDENCE", "APARTMENT", "SIDEWALK", "OTHER",
            "PARKING LOT", "SCHOOL", "ALLEY", "RESIDENCE-GARAGE"]
    yrs = [2020, 2021, 2022, 2023, 2024, 2025]
    samples = []
    for _ in range(rows):
        ct = random.choice(list(arrest_p_by_type))
        h = random.randint(0, 23)
        is_domestic = random.random() < 0.15
        p = arrest_p_by_type[ct] + (0.20 if is_domestic else 0)
        if 2 <= h <= 5:
            p -= 0.10
        p = max(0.01, min(0.99, p))
        samples.append(Row(
            District=random.randint(1, 25),
            **{"Primary Type": ct},
            **{"Location Description": random.choice(locs)},
            Year=random.choice(yrs),
            Hour=h,
            Domestic_str=str(is_domestic).lower(),
            Arrest=random.random() < p,
            label=int(random.random() < p),
        ))
    return spark.createDataFrame(samples)


crimes_df = _load_real_data() if runtime == "cluster" else _load_synthetic_data()
crimes_df.cache()
print("Row count:", f"{crimes_df.count():,}")
crimes_df.printSchema()
crimes_df.show(3, truncate=False)
""")


# ---------- Phase A ----------
md("---\n# Phase A — DataFrame analytics on the full dataset")


md(f"""## Task 1 — Crime type distribution
*Author: {ABDULRAHMAN}*

DataFrame `groupBy` + descending count.""")

code(f"""# Task 1 — author: {ABDULRAHMAN}
crime_type_counts = (crimes_df
                     .groupBy("Primary Type")
                     .agg(fn.count(fn.lit(1)).alias("incident_count"))
                     .orderBy(fn.col("incident_count").desc()))
crime_type_counts.show(10, truncate=False)
""")


md(f"""## Task 2 — Location hotspots (Spark SQL)
*Author: {ABDULMOHSEN}*

Switch to SQL via `createOrReplaceTempView`.""")

code(f"""# Task 2 — author: {ABDULMOHSEN}
crimes_df.createOrReplaceTempView("crime_records")

location_hotspots = spark.sql(\"\"\"
    SELECT  `Location Description` AS hotspot,
            COUNT(*)               AS records
      FROM  crime_records
     WHERE  `Location Description` IS NOT NULL
     GROUP  BY `Location Description`
     ORDER  BY records DESC
     LIMIT  10
\"\"\")
location_hotspots.show(truncate=False)
""")


md(f"""## Task 3 — Year trend
*Author: {SAUD}*

Yearly counts; matplotlib chart in local mode.""")

code(f"""# Task 3 — author: {SAUD}
yearly_counts = (crimes_df
                 .groupBy("Year")
                 .agg(fn.count(fn.lit(1)).alias("incidents"))
                 .orderBy("Year"))
yearly_counts.show(30)
""")

code(f"""# Task 3 chart — author: {SAUD}
if runtime == "local":
    import matplotlib.pyplot as plt
    pdf = yearly_counts.toPandas().dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(pdf["Year"].astype(int), pdf["incidents"], color="#3a6ea5")
    ax.set_xlabel("Year")
    ax.set_ylabel("Incident count")
    ax.set_title("Chicago crime incidents per year")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/year_distribution.png", dpi=120)
    plt.show()
else:
    print("Cluster mode — printed table is the deliverable.")
""")


md(f"""## Task 4 — Arrest rate analysis
*Author: {FERAS}*

Overall rate plus the per-crime-type breakdown.""")

code(f"""# Task 4 — author: {FERAS}
total_records   = crimes_df.count()
arrest_records  = crimes_df.filter(fn.col("Arrest") == True).count()
arrest_rate_pct = arrest_records / total_records * 100
print(f"Overall arrest rate: {{arrest_records:,}} / {{total_records:,}} = {{arrest_rate_pct:.2f}}%")

per_crime_rates = (crimes_df
                   .groupBy("Primary Type")
                   .agg(fn.count(fn.lit(1)).alias("records"),
                        fn.avg(fn.col("label").cast("double")).alias("arrest_rate"))
                   .filter(fn.col("records") >= 100)
                   .orderBy(fn.col("arrest_rate").desc()))
print("Top arrest rates by crime type (min 100 records):")
per_crime_rates.show(15, truncate=False)
""")


# ---------- Phase B ----------
md("""---
# Phase B — MLlib arrest predictor (5% sample per spec)

Per the May 2026 update, Phase B runs on a 5% sample. Locally that's a no-op on the
10K synthetic data. On the cluster the 5% sample reduces 793,072 rows to ~39,654.""")

code("""# 5% sample, seed=42 — applied before any feature engineering
ml_input = crimes_df.sample(fraction=0.05, seed=42)
print(f"Phase B working set: {ml_input.count():,} rows  (5% sample, seed=42)")
""")


md(f"""## Task 5 — Feature pipeline
*Author: {KHALID}*

`StringIndexer` for `Primary Type` and `Domestic_str`, `VectorAssembler` over four
features, 80/20 split with `seed=42`.""")

code(f"""# Task 5 — author: {KHALID}
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler

if "Domestic_str" not in ml_input.columns:
    ml_input = ml_input.withColumn("Domestic_str",
                                   fn.col("Domestic").cast(StringType()))

crime_label_idx    = StringIndexer(inputCol="Primary Type",
                                   outputCol="crime_idx",
                                   handleInvalid="skip")
domestic_label_idx = StringIndexer(inputCol="Domestic_str",
                                   outputCol="dom_idx",
                                   handleInvalid="skip")
feature_assembler  = VectorAssembler(
    inputCols=["Hour", "crime_idx", "District", "dom_idx"],
    outputCol="feat_vec",
)

train_set, test_set = ml_input.randomSplit([0.8, 0.2], seed=42)
train_set.cache()
test_set.cache()
print(f"Train rows: {{train_set.count():,}}   Test rows: {{test_set.count():,}}")

# Visualise what the assembled feature column looks like for 5 rows
preview_pipe = Pipeline(stages=[crime_label_idx, domestic_label_idx, feature_assembler]).fit(train_set)
preview_pipe.transform(train_set).select(
    "Primary Type", "crime_idx",
    "Hour", "District",
    "Domestic_str", "dom_idx",
    "feat_vec", "label",
).show(5, truncate=False)
print("Vector layout: [Hour, crime_idx, District, dom_idx]")
""")


md(f"""## Task 6 — Train and evaluate three classifiers
*Author: {KHALID}*

Logistic Regression (maxIter=100, regParam=0.01), Random Forest (numTrees=100,
maxDepth=5), GBT (maxIter=50, maxDepth=5). `maxBins=64` because Primary Type has
more than 32 categories on the cluster.""")

code(f"""# Task 6 helpers — author: {KHALID}
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator,
)

binary_evaluator = BinaryClassificationEvaluator(labelCol="label")
multi_evaluator  = MulticlassClassificationEvaluator(labelCol="label",
                                                     predictionCol="prediction")


def eval_metrics(predictions):
    measure = lambda metric: multi_evaluator.evaluate(predictions,
                                                      {{multi_evaluator.metricName: metric}})
    return {{
        "AUC":       binary_evaluator.evaluate(predictions),
        "Accuracy":  measure("accuracy"),
        "F1":        measure("f1"),
        "Precision": measure("weightedPrecision"),
        "Recall":    measure("weightedRecall"),
    }}


def confusion_counts(predictions):
    grid = {{(int(r["label"]), int(r["prediction"])): r["count"]
            for r in predictions.groupBy("label", "prediction").count().collect()}}
    return (grid.get((0, 0), 0), grid.get((0, 1), 0),
            grid.get((1, 0), 0), grid.get((1, 1), 0))


classifier_specs = [
    ("LogisticRegression",
     LogisticRegression(featuresCol="feat_vec", labelCol="label",
                        maxIter=100, regParam=0.01)),
    ("RandomForest",
     RandomForestClassifier(featuresCol="feat_vec", labelCol="label",
                            numTrees=100, maxDepth=5,
                            maxBins=64, seed=42)),
    ("GBT",
     GBTClassifier(featuresCol="feat_vec", labelCol="label",
                   maxIter=50, maxDepth=5,
                   maxBins=64, seed=42)),
]
""")

code(f"""# Task 6 training loop — author: {KHALID}
training_results = []
fitted_random_forest = None
for tag, learner in classifier_specs:
    print(f"\\n>>> training {{tag}}")
    pipeline = Pipeline(stages=[crime_label_idx, domestic_label_idx,
                                feature_assembler, learner])
    started_at = time.time()
    fitted = pipeline.fit(train_set)
    secs = time.time() - started_at
    preds = fitted.transform(test_set)
    metrics = eval_metrics(preds)
    cm = confusion_counts(preds)
    training_results.append((tag, fitted, metrics, cm, secs))
    print(f"  AUC       {{metrics['AUC']:.4f}}")
    print(f"  Accuracy  {{metrics['Accuracy']:.4f}}")
    print(f"  F1        {{metrics['F1']:.4f}}")
    print(f"  Precision {{metrics['Precision']:.4f}}")
    print(f"  Recall    {{metrics['Recall']:.4f}}")
    print(f"  Train(s)  {{secs:.1f}}")
    print(f"  CM (TN, FP, FN, TP) = {{cm}}")
    if tag == "RandomForest":
        fitted_random_forest = fitted.stages[-1]

# Summary table
print("\\n" + "=" * 76)
print(f"{{'metric':<11}}{{'Logistic':>14}}{{'RandomForest':>16}}{{'GBT':>14}}")
print("-" * 76)
m_lr, m_rf, m_gbt = [r[2] for r in training_results]
for k in ("AUC", "Accuracy", "F1", "Precision", "Recall"):
    print(f"{{k:<11}}{{m_lr[k]:>14.4f}}{{m_rf[k]:>16.4f}}{{m_gbt[k]:>14.4f}}")
print(f"{{'Train(s)':<11}}{{training_results[0][4]:>14.1f}}{{training_results[1][4]:>16.1f}}{{training_results[2][4]:>14.1f}}")
print("=" * 76)
top = max(training_results, key=lambda x: x[2]["AUC"])
print(f"Top model by AUC: {{top[0]}} ({{top[2]['AUC']:.4f}})")
""")


md(f"""## Task 7 — Random Forest feature importances
*Author: {ABDULMOHSEN}*

Importances tell us which feature drives most of the splits in the forest.""")

code(f"""# Task 7 — author: {ABDULMOHSEN}
layout_for_rf = ["Hour", "crime_idx", "District", "dom_idx"]
rf_importances = fitted_random_forest.featureImportances.toArray()

print("Random Forest feature importances:")
for feat, imp in sorted(zip(layout_for_rf, rf_importances), key=lambda kv: -kv[1]):
    print(f"  {{feat:<12}} {{imp:.4f}}  {{('|' * int(round(imp * 50)))}}")
""")


md("""**Reading the importances.** The crime-type index dominates because the per-crime
arrest-rate distribution from Task 4 is itself dominated by crime type
(NARCOTICS ≈ 99% vs THEFT ≈ 14%). Once a tree splits on the crime-type index it has
most of its answer. Logistic Regression underperforms the tree models because it
treats `crime_idx` as a numeric feature and fits a linear coefficient — implying a
meaningless ordering between crime types. Trees split on individual values of the
index so the ordering does not matter.""")


md("""---
## Cleanup""")

code("""spark.stop()""")


# ---------- Write the notebook ----------
nb = {
    "cells": _cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path(OUTPUT).write_text(json.dumps(nb, indent=1))
print(f"wrote {OUTPUT} ({len(_cells)} cells)")

"""SE446 Milestone 2 — Big Data Guys

Phase B (Tasks 5-7): standalone Spark MLlib pipeline for the spark-submit deliverable.

Authors:
    Task 5 — Khalid Aleisa (230525)
    Task 6 — Khalid Aleisa (230525)
    Task 7 — Abdulmohsen Binkhamis (230241)

The May 2026 spec update applies:
    - Task 8 (CrossValidator) is omitted.
    - Phase B trains on a 5% sample (df.sample(fraction=0.05, seed=42)).

Submit via:
    spark-submit \\
        --master yarn --deploy-mode cluster \\
        --num-executors 2 --executor-memory 1g --executor-cores 1 \\
        chicago_arrest_pipeline.py
"""
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as fn
from pyspark.sql.types import IntegerType, StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator,
)


HDFS_DATASET_PATH = "hdfs:///data/chicago_crimes.csv"


def get_session() -> SparkSession:
    return (SparkSession.builder
            .appName("M2_BigDataGuys_chicago_arrest_pipeline")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())


def load_dataset(session: SparkSession):
    rows_raw = session.read.csv(HDFS_DATASET_PATH, header=True, inferSchema=True)
    rows = (rows_raw
            .withColumn("Hour",
                        fn.hour(fn.to_timestamp(fn.col("Date"),
                                                "MM/dd/yyyy hh:mm:ss a")))
            .withColumn("label",        fn.col("Arrest").cast(IntegerType()))
            .withColumn("Domestic_str", fn.col("Domestic").cast(StringType())))
    return rows.dropna(subset=["District", "Primary Type",
                               "Hour", "Domestic_str", "label"])


def metrics_dict(predictions, binary_eval, multi_eval):
    fetch = lambda metric: multi_eval.evaluate(predictions,
                                               {multi_eval.metricName: metric})
    return {
        "AUC":       binary_eval.evaluate(predictions),
        "Accuracy":  fetch("accuracy"),
        "F1":        fetch("f1"),
        "Precision": fetch("weightedPrecision"),
        "Recall":    fetch("weightedRecall"),
    }


def confusion_quad(predictions):
    grid = {(int(r["label"]), int(r["prediction"])): r["count"]
            for r in predictions.groupBy("label", "prediction").count().collect()}
    return (grid.get((0, 0), 0), grid.get((0, 1), 0),
            grid.get((1, 0), 0), grid.get((1, 1), 0))


def main():
    spark = get_session()
    print("Spark version: ", spark.version)
    print("Master:        ", spark.sparkContext.master)

    crimes = load_dataset(spark)
    print("Full dataset rows:", f"{crimes.count():,}")

    # ----- Task 5 (Khalid): pipeline + 5% sample -----
    sample = crimes.sample(fraction=0.05, seed=42)
    print("Phase B sample:", f"{sample.count():,} rows  (5%, seed=42)")

    crime_idx_stage = StringIndexer(inputCol="Primary Type",
                                    outputCol="crime_idx",
                                    handleInvalid="skip")
    dom_idx_stage   = StringIndexer(inputCol="Domestic_str",
                                    outputCol="dom_idx",
                                    handleInvalid="skip")
    feature_stage   = VectorAssembler(
        inputCols=["Hour", "crime_idx", "District", "dom_idx"],
        outputCol="feat_vec",
    )

    train_set, test_set = sample.randomSplit([0.8, 0.2], seed=42)
    train_set.cache()
    test_set.cache()
    print("Train rows:", f"{train_set.count():,}", " | Test rows:", f"{test_set.count():,}")

    binary_eval = BinaryClassificationEvaluator(labelCol="label")
    multi_eval  = MulticlassClassificationEvaluator(labelCol="label",
                                                    predictionCol="prediction")

    learners = [
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

    rf_inner = None
    summary  = []
    for tag, learner in learners:
        pipeline = Pipeline(stages=[crime_idx_stage, dom_idx_stage,
                                    feature_stage, learner])
        t0 = time.time()
        fitted = pipeline.fit(train_set)
        secs = time.time() - t0
        preds = fitted.transform(test_set)
        m = metrics_dict(preds, binary_eval, multi_eval)
        cm = confusion_quad(preds)
        summary.append((tag, secs, m, cm))
        print(f"\n>>> {tag}")
        for key, value in m.items():
            print(f"  {key:<10}{value:.4f}")
        print(f"  Train(s)  {secs:.1f}")
        print(f"  CM (TN,FP,FN,TP) = {cm}")
        if tag == "RandomForest":
            rf_inner = fitted.stages[-1]

    print("\n" + "=" * 76)
    print(f"{'metric':<11}{'Logistic':>14}{'RandomForest':>16}{'GBT':>14}")
    print("-" * 76)
    for key in ("AUC", "Accuracy", "F1", "Precision", "Recall"):
        print(f"{key:<11}{summary[0][2][key]:>14.4f}{summary[1][2][key]:>16.4f}{summary[2][2][key]:>14.4f}")
    print(f"{'Train(s)':<11}{summary[0][1]:>14.1f}{summary[1][1]:>16.1f}{summary[2][1]:>14.1f}")
    print("=" * 76)
    top = max(summary, key=lambda r: r[2]["AUC"])
    print("Top model by AUC:", top[0], f"({top[2]['AUC']:.4f})")

    # ----- Task 7 (Abdulmohsen): RF feature importances -----
    print("\n--- Random Forest feature importances ---")
    layout = ["Hour", "crime_idx", "District", "dom_idx"]
    for feat, imp in sorted(zip(layout, rf_inner.featureImportances.toArray()),
                            key=lambda kv: -kv[1]):
        print(f"  {feat:<12}{imp:.4f}  {'|' * int(round(imp * 50))}")

    spark.stop()


if __name__ == "__main__":
    main()

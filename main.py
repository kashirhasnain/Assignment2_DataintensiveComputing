import os
os.environ["PYSPARK_PYTHON"] = r"C:\Users\hasna\AppData\Local\Programs\Python\Python311\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\hasna\AppData\Local\Programs\Python\Python311\python.exe"

import html
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    CountVectorizer, IDF, StopWordsRemover, ChiSqSelector,
    StringIndexer, RegexTokenizer, Normalizer,
from pyspark.ml.feature import {
    CountVectorizer, IDF, StopWordsRemover, ChiSqSelector, StringIndexer, RegexTokenizer
)
from pyspark.ml.classification import LinearSVC, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# Part 1 delimiter
DELIM_RE = re.compile(r"[ \t\d()\[\]{}\.\!\?,;:\+=_\"'`~#@&\*%€$§\\/ -]+")

# Part 2/3 delimiter (as regex string for Spark split)
DELIM_PAT = r"[ \t\d()\[\]{}\.\!\?,;:\+=\-_\"'`~#@&\*%€$§\\/]+"


# ─── Part 1 helpers ───────────────────────────────────────────────────────────

def load_stopwords(path: str, sc: SparkContext) -> set:
    if path.startswith("hdfs://"):
        return {line.strip().lower() for line in sc.textFile(path).collect() if line.strip()}
    local_path = Path(path)
    with local_path.open("r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def tokenize(text: str, stopwords: set) -> list:
    text = html.unescape(text).lower()
    tokens = [tok for tok in DELIM_RE.split(text) if tok]
    return [tok for tok in tokens if tok not in stopwords and len(tok) > 1]


def process_line(line: str, stopwords: set) -> list:
    line = line.strip()
    if not line:
        return []
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return []

    category = record.get("category")
    if not isinstance(category, str) or not category:
        return []

    parts = []
    summary = record.get("summary")
    review_text = record.get("reviewText")
    if isinstance(summary, str) and summary.strip():
        parts.append(summary)
    if isinstance(review_text, str) and review_text.strip():
        parts.append(review_text)

    tokens = tokenize(" ".join(parts), stopwords)
    unique_terms = {t for t in tokens if t}

    outputs = [("TOTAL", 1), (f"CAT\t{category}", 1)]
    for term in unique_terms:
        outputs.append((f"TERM\t{term}", 1))
        outputs.append((f"CATTERM\t{category}\t{term}", 1))
    return outputs


def compute_chi2(
    category: str,
    cat_term_items: list,
    total_docs: int,
    category_docs: dict,
    term_docs: dict,
    top_k: int,
) -> tuple:
    rows = []
    cat_docs = category_docs.get(category, 0)
    for term, a in cat_term_items:
        b = cat_docs - a
        c = term_docs.get(term, 0) - a
        d = total_docs - (a + b + c)
        denom = (a + c) * (b + d) * (a + b) * (c + d)
        chi2 = 0.0 if denom == 0 else total_docs * ((a * d) - (b * c)) ** 2 / denom
        rows.append((chi2, term))
    rows.sort(key=lambda item: item[0], reverse=True)
    return category, rows[:top_k]


# ─── Part 2 ───────────────────────────────────────────────────────────────────

def part2(input_file: str, stopwords_file: str) -> None:
    spark = SparkSession.builder.appName("ChiSquareDataFrame").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    stopwords = [l.strip().lower() for l in Path(stopwords_file).open(encoding="utf-8") if l.strip()]

    df = (
        spark.read.json(input_file)
        .filter(F.col("category").isNotNull() & (F.trim(F.col("category")) != ""))
        .withColumn("raw_text", F.concat_ws(" ",
            F.coalesce(F.col("summary"),    F.lit("")),
            F.coalesce(F.col("reviewText"), F.lit("")),
        ))
        .filter(F.trim(F.col("raw_text")) != "")
        .select("category", "raw_text")
        .cache()
    )

    pipeline = Pipeline(stages=[
        RegexTokenizer(
            inputCol="raw_text", outputCol="tokens_raw",
            pattern=DELIM_PAT, minTokenLength=2, toLowercase=True, gaps=True,
        ),
        StopWordsRemover(
            inputCol="tokens_raw", outputCol="tokens",
            stopWords=stopwords, caseSensitive=False,
        ),
        StringIndexer(
            inputCol="category", outputCol="label",
            handleInvalid="keep", stringOrderType="frequencyDesc",
        ),
        CountVectorizer(
            inputCol="tokens", outputCol="tf",
            minDF=2.0, maxDF=0.95, vocabSize=50_000,
        ),
        IDF(inputCol="tf", outputCol="tfidf", minDocFreq=2),
        ChiSqSelector(
            featuresCol="tfidf", outputCol="selected",
            labelCol="label", selectorType="numTopFeatures",
            numTopFeatures=2000,
        ),
    ])

    model = pipeline.fit(df)

    vocab          = model.stages[3].vocabulary
    selected_idx   = set(model.stages[5].selectedFeatures)
    selected_terms = sorted(vocab[i] for i in selected_idx)

    out = Path("output")
    out.mkdir(exist_ok=True)
    out.joinpath("output_ds.txt").write_text(" ".join(selected_terms) + "\n", encoding="utf-8")
    print(f"✅ Part 2: {len(selected_terms)} terms written to output/output_ds.txt")

    df.unpersist()
    spark.stop()
     # ── Part 3────────────────────────────────────────────────────────────────
  
spark = SparkSession.builder \
    .appName("Task3") \
    .getOrCreate()


df = spark.read.csv(
   "/Users/qaisarzada/task3/R_small.csv",
    header=True,
    inferSchema=True
)


df = df.withColumn(
    "text",
    lower(col("text"))
)


label_indexer = StringIndexer(
    inputCol="label",
    outputCol="labelIndex"
)

tokenizer = Tokenizer(
    inputCol="text",
    outputCol="tokens"
)


remover = StopWordsRemover(
    inputCol="tokens",
    outputCol="filtered"
)


cv = CountVectorizer(
    inputCol="filtered",
    outputCol="rawFeatures"
)


idf = IDF(
    inputCol="rawFeatures",
    outputCol="features"
)


selector = ChiSqSelector(
    numTopFeatures=2000,
    featuresCol="features",
    outputCol="selectedFeatures",
    labelCol="labelIndex"
)


pipeline = Pipeline(stages=[
    label_indexer,
    tokenizer,
    remover,
    cv,
    idf,
    selector
])


model = pipeline.fit(df)


result = model.transform(df)


result.select(
    "label",
    "selectedFeatures"
).show(truncate=False)


result.select(
    "label",
    "selectedFeatures"
).write.mode("overwrite").json("output_ds")

spark.stop()


# ─── Part 3 ───────────────────────────────────────────────────────────────────

def part3(input_file: str, stopwords_file: str) -> None:
    spark = SparkSession.builder.appName("TextClassification").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    stopwords = [l.strip().lower() for l in Path(stopwords_file).open(encoding="utf-8") if l.strip()]

    df = (
        spark.read.json(input_file)
        .filter(F.col("category").isNotNull() & (F.trim(F.col("category")) != ""))
        .withColumn("raw_text", F.concat_ws(" ",
            F.coalesce(F.col("summary"),    F.lit("")),
            F.coalesce(F.col("reviewText"), F.lit("")),
        ))
        .filter(F.trim(F.col("raw_text")) != "")
        .select("category", "raw_text")
        .cache()
    )

    # ── Reproducible train / validation / test split ──────────────────────────
    train_df, val_df, test_df = df.randomSplit([0.7, 0.15, 0.15], seed=42)
    train_df.cache()
    val_df.cache()
    test_df.cache()

    # ── Pipeline stages ───────────────────────────────────────────────────────
    tokenizer = RegexTokenizer(
        inputCol="raw_text", outputCol="tokens_raw",
        pattern=DELIM_PAT, minTokenLength=2, toLowercase=True, gaps=True,
    )
    remover = StopWordsRemover(
        inputCol="tokens_raw", outputCol="tokens",
        stopWords=stopwords, caseSensitive=False,
    )
    indexer = StringIndexer(
        inputCol="category", outputCol="label",
        handleInvalid="keep", stringOrderType="frequencyDesc",
    )
    cv  = CountVectorizer(inputCol="tokens",  outputCol="tf",     minDF=2.0, maxDF=0.95, vocabSize=50_000)
    idf = IDF(             inputCol="tf",      outputCol="tfidf",  minDocFreq=2)

    # Stage index 5 — dimensionality varies in grid
    selector = ChiSqSelector(
        featuresCol="tfidf", outputCol="selected",
        labelCol="label", selectorType="numTopFeatures",
        numTopFeatures=2000,
    )

    # Stage index 6 — L2 normalisation before classifier (required)
    normalizer = Normalizer(inputCol="selected", outputCol="features", p=2.0)

    # Binary SVM wrapped in One-vs-Rest for multiclass
    svm = LinearSVC(featuresCol="features", labelCol="label")
    ovr = OneVsRest(classifier=svm, featuresCol="features", labelCol="label")

    pipeline = Pipeline(stages=[
        tokenizer,   # 0
        remover,     # 1
        indexer,     # 2
        cv,          # 3
        idf,         # 4
        selector,    # 5
        normalizer,  # 6
        ovr,         # 7
    ])

    
    param_grid = [
        {
            "numTopFeatures": ntf,
            "regParam":       reg,
            "standardization": std,
            "maxIter":        itr,
        }
        for ntf in [2000, 500]
        for reg in [0.01, 0.1, 1.0]
        for std in [True, False]
        for itr in [50, 100]
    ]

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    print(f"\n── Part 3: grid search over {len(param_grid)} combinations ──")

    best_f1     = -1.0
    best_model  = None
    best_params = {}

    for i, params in enumerate(param_grid):
        print(f"\n[{i+1}/{len(param_grid)}] "
              f"features={params['numTopFeatures']}  "
              f"reg={params['regParam']}  "
              f"std={params['standardization']}  "
              f"maxIter={params['maxIter']}")

        t0 = time.perf_counter()

        # Apply params to the relevant stages before fitting
        pipeline.getStages()[5].setNumTopFeatures(params["numTopFeatures"])
        svm.setRegParam(params["regParam"])
        svm.setStandardization(params["standardization"])
        svm.setMaxIter(params["maxIter"])

        model      = pipeline.fit(train_df)
        val_preds  = model.transform(val_df)
        f1         = evaluator.evaluate(val_preds)
        elapsed    = time.perf_counter() - t0

        print(f"    Val F1 = {f1:.4f}  ({elapsed:.1f}s)")

        if f1 > best_f1:
            best_f1     = f1
            best_model  = model
            best_params = dict(params)

    print(f"\n✅ Best validation F1 : {best_f1:.4f}")
    print(f"   Best params        : {best_params}")

    # ── Final evaluation on held-out test set ─────────────────────────────────
    test_preds = best_model.transform(test_df)
    test_f1    = evaluator.evaluate(test_preds)
    print(f"✅ Test F1 (best model): {test_f1:.4f}")

    for metric in ["accuracy", "weightedPrecision", "weightedRecall"]:
        val = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName=metric
        ).evaluate(test_preds)
        print(f"   Test {metric:20s}: {val:.4f}")

    # ── Write results ─────────────────────────────────────────────────────────
    out = Path("output")
    out.mkdir(exist_ok=True)

    lines = [
        f"Best params        : {best_params}",
        f"Best validation F1 : {best_f1:.4f}",
        f"Test F1            : {test_f1:.4f}",
    ]
    out.joinpath("output_classification.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("✅ Part 3: results written to output/output_classification.txt")

    train_df.unpersist()
    val_df.unpersist()
    test_df.unpersist()
    df.unpersist()
    spark.stop()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    started = time.perf_counter()

    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    else:
        input_file = os.path.join(os.getcwd(), "Assignment_1_Assets", "reviews_devset.json")

    if len(sys.argv) >= 3:
        stopwords_file = sys.argv[2]
    else:
        stopwords_file = os.path.join("Assignment_1_Assets", "stopwords.txt")

    if "://" not in input_file:
        input_file = f"file:///{os.path.abspath(input_file)}".replace("\\", "/")

    top_k = 75

    # ── Part 1 ────────────────────────────────────────────────────────────────
    conf = SparkConf().setAppName("ChiSquareRDD").setMaster("local[*]")
    sc   = SparkContext(conf=conf)

    stopwords       = load_stopwords(stopwords_file, sc)
    stopwords_bcast = sc.broadcast(stopwords)

    lines      = sc.textFile(input_file)
    counts_rdd = (
        lines
        .flatMap(lambda line: process_line(line, stopwords_bcast.value))
        .reduceByKey(lambda a, b: a + b)
    )

    counts       = counts_rdd.collectAsMap()
    total_docs   = counts.get("TOTAL", 0)
    category_docs = {k.split("\t")[1]: v for k, v in counts.items() if k.startswith("CAT\t")}
    term_docs     = {k.split("\t")[1]: v for k, v in counts.items() if k.startswith("TERM\t")}

    cat_term_docs = defaultdict(list)
    for k, v in counts.items():
        if k.startswith("CATTERM\t"):
            _, category, term = k.split("\t")
            cat_term_docs[category].append((term, v))

    categories_rdd = sc.parallelize(list(cat_term_docs.items()))
    top_k_per_cat  = categories_rdd.map(
        lambda item: compute_chi2(
            item[0], item[1], total_docs, category_docs, term_docs, top_k
        )
    ).collect()

    top_k_per_cat.sort(key=lambda x: x[0])
    output_lines = []
    merged_terms = set()
    for category, top_items in top_k_per_cat:
        parts = [category]
        for chi2, term in top_items:
            merged_terms.add(term)
            parts.append(f"{term}:{chi2:.6f}")
        output_lines.append(" ".join(parts))

    output_lines.append(" ".join(sorted(merged_terms)))
    result = "\n".join(output_lines)

    out = Path("output")
    out.mkdir(exist_ok=True)
    out.joinpath("output_rdd.txt").write_text(result + "\n", encoding="utf-8")
    print(f"✅ Part 1: {len(merged_terms)} terms written to output/output_rdd.txt")

    elapsed = time.perf_counter() - started
    hours   = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    if hours > 0:
        print(f"Elapsed: {hours}h {minutes}m {seconds}s")
    elif minutes > 0:
        print(f"Elapsed: {minutes}m {seconds}s")
    else:
        print(f"Elapsed: {seconds}s")

    sc.stop()

    # ── Part 2 ────────────────────────────────────────────────────────────────
    part2(input_file, stopwords_file)

    # ── Part 3 ────────────────────────────────────────────────────────────────
    part3(input_file, stopwords_file)


if __name__ == "__main__":
    main()

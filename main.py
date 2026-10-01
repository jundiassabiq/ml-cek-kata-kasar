"""Train and use Indonesian offensive-language classifiers from the CLI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "bad_sentences.csv"


def load_data(path: Path) -> pd.DataFrame:
    """Read, validate, and lightly clean the labeled CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_csv(path)
    required = {"sentence", "contains_bad_word"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(sorted(missing))}")

    df = df[["sentence", "contains_bad_word"]].dropna().copy()
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df["contains_bad_word"] = pd.to_numeric(df["contains_bad_word"], errors="coerce")
    df = df.dropna(subset=["contains_bad_word"])
    df = df[df["sentence"].ne("")]
    if not df["contains_bad_word"].isin([0, 1]).all():
        raise ValueError("Label contains_bad_word harus hanya berisi 0 atau 1.")
    df["contains_bad_word"] = df["contains_bad_word"].astype(int)

    before = len(df)
    df = df[df["sentence"].str.contains(r"\banjing\b", case=False, regex=True)].copy()
    print(f"Filter fokus anjing: {len(df)} dari {before} baris mengandung kata anjing.")
    df = df.drop_duplicates(subset=["sentence"], keep="first").reset_index(drop=True)
    print(f"Data siap: {len(df)} baris setelah duplikat dihapus.")
    print("Distribusi label:")
    for label, count in df["contains_bad_word"].value_counts().sort_index().items():
        print(f"  {label} ({'kasar' if label == 1 else 'tidak kasar'}): {count}")
    if df["contains_bad_word"].nunique() != 2:
        raise ValueError("Dataset harus memiliki contoh untuk label 0 dan label 1.")
    return df


def make_models() -> dict[str, Pipeline]:
    """Return four experiments: word uni/bi-grams, char n-grams, and both."""
    word_unigram = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), lowercase=True,
                                   strip_accents="unicode", sublinear_tf=True)
    word_unibigram = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True,
                                     strip_accents="unicode", sublinear_tf=True)
    char_ngrams = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=True,
                                  strip_accents="unicode", sublinear_tf=True,
                                  min_df=1, max_features=100_000)
    combined = FeatureUnion([("word", TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), lowercase=True,
        strip_accents="unicode", sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                 lowercase=True, strip_accents="unicode",
                                 sublinear_tf=True, min_df=1, max_features=100_000))])
    configs = {
        "Kata unigram": word_unigram,
        "Kata unigram + bigram": word_unibigram,
        "Karakter 2-5 gram": char_ngrams,
        "Gabungan kata + karakter": combined,
    }
    return {name: Pipeline([("tfidf", vectorizer),
                            ("classifier", LogisticRegression(max_iter=2000,
                                                              class_weight="balanced",
                                                              random_state=42))])
            for name, vectorizer in configs.items()}


def evaluate(data_path: Path) -> dict[str, Pipeline]:
    df = load_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        df["sentence"], df["contains_bad_word"], test_size=0.2,
        random_state=42, stratify=df["contains_bad_word"])
    print(f"\nPembagian data: {len(x_train)} latih, {len(x_test)} evaluasi (80:20).")
    models = make_models()
    for name, model in models.items():
        print(f"\n{'=' * 68}\nModel: {name}\n{'=' * 68}")
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        print(classification_report(y_test, predictions, labels=[0, 1],
                                    target_names=["tidak kasar", "kasar"],
                                    zero_division=0, digits=3))
        print("Confusion matrix (baris=aktual, kolom=prediksi; urutan 0,1):")
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Cek satu kalimat untuk penggunaan kasar kata anjing.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Lokasi CSV dengan kolom sentence dan contains_bad_word")
    args = parser.parse_args()
    try:
        df = load_data(args.data)
        model = make_models()["Gabungan kata + karakter"]
        model.fit(df["sentence"], df["contains_bad_word"])
        sentence = input("Masukkan satu kalimat: ").strip()
        if not sentence:
            print("Kalimat kosong. Tidak ada prediksi.")
            return
        predicted = int(model.predict([sentence])[0])
        probability = model.predict_proba([sentence])[0]
        confidence = probability[predicted] * 100
        label = "KASAR" if predicted else "TIDAK KASAR"
        print(f"Hasil: {label} (keyakinan model {confidence:.1f}%)")
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

"""
CVE summary에서 n-gram 상위 K개를 피처 열로 붙인다.

meaning 열 값: 기본은 해당 summary 안에서 n-gram 등장 횟수(정수, 0·1·2…).
  --binary-meanings 를 주면 존재 여부만 0/1.

기본: CVE_summary_separate/meaning_lexicon.json 의 상투구·기술어 목록으로 필터,
     순위는 (필터 통과 n-gram 중) 문서 빈도 df 내림차순.

--no-filter-boilerplate: 문서 빈도(df)만으로 상위 K개 (예전 방식).

--add-meta-columns: Timestamp·Anomaly_ID만 추가, 라벨은 cvss 열 그대로.

기본 출력: cve_with_meaning70.csv, cve_summary_meaning_top70.json (--top-k 기본 70).

boilerplate_tokens 규칙(토큰 단위, tech_lexicon·부분 문자열은 별도):
  - 구절을 단어로 쪼갠 뒤, 모든 단어가 boilerplate_tokens 에만 속하면 → 상투구.
  - 하나라도 그 집합에 없으면 → 상투구 아님(살림). 예: unigram memory 만 있으면 상투구,
    memory corruption 은 corruption 이 보통 boilerplate_tokens 에 없으므로 살아남음.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# `python severity/build_cve_summary_meaning_features.py` 시에도 `severity.*` 임포트 가능하도록
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from severity.severity_schema import ANOMALY_ID_COL, TIMESTAMP_COL

SEVERITY_DIR = Path(__file__).resolve().parent
# 로컬 CVE 모음(레포 내): severity/CVE_summary_separate/cve-datasets/
_CVE_SEP = SEVERITY_DIR / "CVE_summary_separate"
_DEFAULT_REL = _CVE_SEP / "cve-datasets"
DEFAULT_INPUT = _DEFAULT_REL / "cve.csv"
DEFAULT_OUTPUT = _DEFAULT_REL / "cve_with_meaning70.csv"
DEFAULT_VOCAB_JSON = _DEFAULT_REL / "cve_summary_meaning_top70.json"
DEFAULT_LEXICON_JSON = _CVE_SEP / "meaning_lexicon.json"


@dataclass(frozen=True)
class MeaningLexicon:
    boilerplate_substrings: tuple[str, ...]
    boilerplate_tokens: frozenset[str]
    tech_lexicon: frozenset[str]


def load_meaning_lexicon(path: Path) -> MeaningLexicon:
    if not path.is_file():
        raise FileNotFoundError(f"meaning lexicon 없음: {path.resolve()}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return MeaningLexicon(
        boilerplate_substrings=tuple(raw["boilerplate_substrings"]),
        boilerplate_tokens=frozenset(raw["boilerplate_tokens"]),
        tech_lexicon=frozenset(raw["tech_lexicon"]),
    )


def _is_boilerplate_ngram(term: str, lex: MeaningLexicon) -> bool:
    """n-gram 문자열이 상투구면 True.

    순서: 빈 문자열 → boilerplate_substrings 부분 일치 → 토큰 분리 →
    토큰 중 하나라도 tech_lexicon 이면 상투구 아님 →
    그렇지 않으면 **모든** 토큰이 boilerplate_tokens 에 있을 때만 상투구(all 규칙).

    범용어(memory, file, …)만 boilerplate_tokens 에 넣으면 unigram memory 는 걸러지고
    memory corruption 은 corruption 등 비목록 토큰 때문에 all() 이 깨져 남는다.
    """
    t = term.lower().strip()
    if not t:
        return True
    if any(s in t for s in lex.boilerplate_substrings):
        return True
    words = t.replace("-", " ").split()
    if not words:
        return True
    if any(w in lex.tech_lexicon for w in words):
        return False
    return all(w in lex.boilerplate_tokens for w in words)


def build_meaning_columns(
    texts: list[str],
    *,
    top_k: int,
    ngram_min: int,
    ngram_max: int,
    min_df: int,
    max_df: float,
    filter_boilerplate: bool = True,
    lexicon: MeaningLexicon | None = None,
    *,
    binary_meanings: bool = False,
) -> tuple[np.ndarray, list[str], CountVectorizer, list[int], list[float]]:
    vectorizer = CountVectorizer(
        lowercase=True,
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
        token_pattern=r"(?u)\b[a-z][a-z0-9_\-]{1,}\b",
        max_features=50000,
    )
    X = vectorizer.fit_transform(texts)
    n_docs = X.shape[0]
    doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()

    if not filter_boilerplate:
        order = np.argsort(-doc_freq)
        k = min(top_k, len(order))
        top_idx = order[:k].astype(np.int64)
    else:
        if lexicon is None:
            raise ValueError("filter_boilerplate=True 일 때 lexicon 이 필요합니다.")
        good = np.array([not _is_boilerplate_ngram(str(t), lexicon) for t in terms], dtype=bool)
        if not good.any():
            good = np.ones(len(terms), dtype=bool)
        cand = np.flatnonzero(good)
        # 상투구 제거 후에는 df 내림차순: 희귀 n-gram만 남기는 IDF순은 잡음(sent, hat)에 취약함
        sub_df = doc_freq[cand]
        top_idx = cand[np.argsort(-sub_df)[: min(top_k, len(cand))]]
        if len(top_idx) < top_k:
            print(
                f"[경고] 상투구 제외 후 후보가 {len(top_idx)}개뿐입니다 "
                f"(요청 top_k={top_k}). min-df 낮추거나 max-df 높이면 보강됩니다.",
                flush=True,
            )

    top_terms = [str(terms[i]) for i in top_idx]
    top_doc_freq = doc_freq[top_idx].tolist()
    top_scores = [float(doc_freq[i]) for i in top_idx]
    X_top = X[:, top_idx]
    if binary_meanings:
        X_meaning = (X_top > 0).astype(np.int8).toarray()
    else:
        # CountVectorizer: 문서별 n-gram 등장 횟수(겹치지 않는 슬라이딩 윈도우 카운트)
        X_meaning = np.asarray(X_top.toarray(), dtype=np.int32)
    return X_meaning, top_terms, vectorizer, top_doc_freq, top_scores


def main() -> None:
    p = argparse.ArgumentParser(
        description="CVE summary 상위 의미(n-gram) 피처 생성 (기본: 등장 횟수, 옵션: 0/1)"
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="입력 cve.csv")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="출력 CSV")
    p.add_argument("--vocab-json", type=Path, default=DEFAULT_VOCAB_JSON, help="구절·열이름 매핑 JSON")
    p.add_argument("--top-k", type=int, default=70)
    p.add_argument(
        "--ngram-min",
        type=int,
        default=1,
        help="1이면 tcp, xss 등 단일 토큰 후보 포함",
    )
    p.add_argument("--ngram-max", type=int, default=4)
    p.add_argument(
        "--no-filter-boilerplate",
        action="store_true",
        help="상투구 필터·IDF 순위 끄고, df 상위 K개만 사용",
    )
    p.add_argument(
        "--min-df",
        type=int,
        default=25,
        help="CountVectorizer min_df: n-gram이 최소 이 개수의 문서(summary)에 등장해야 후보에 남김 (정수=문서 개수)",
    )
    p.add_argument(
        "--max-df",
        type=float,
        default=0.65,
        help="CountVectorizer max_df: 전체 문서의 이 비율(0~1)보다 더 많은 문서에 나오면 제외 (너무 흔한 구절 차단)",
    )
    p.add_argument(
        "--add-meta-columns",
        action="store_true",
        help=f"{TIMESTAMP_COL}, {ANOMALY_ID_COL} 추가 (pub_date 기반). 라벨은 cvss 열(숫자) 그대로 사용.",
    )
    p.add_argument(
        "--lexicon-json",
        type=Path,
        default=DEFAULT_LEXICON_JSON,
        help="상투구·기술어 목록 JSON (기본: CVE_summary_separate/meaning_lexicon.json)",
    )
    p.add_argument(
        "--binary-meanings",
        action="store_true",
        help="meaning 열을 0/1만 저장 (등장 횟수 대신 존재 여부)",
    )
    args = p.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8", on_bad_lines="skip")
    if "Unnamed: 0" in df.columns and "cve_id" not in df.columns:
        df = df.rename(columns={"Unnamed: 0": "cve_id"})

    if "summary" not in df.columns:
        raise ValueError("입력 CSV에 'summary' 열이 필요합니다.")

    texts = df["summary"].fillna("").astype(str).tolist()
    filter_bp = not args.no_filter_boilerplate
    lex = load_meaning_lexicon(args.lexicon_json) if filter_bp else None
    X_meaning, top_terms, vec, top_doc_freq, top_scores = build_meaning_columns(
        texts,
        top_k=args.top_k,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_df=args.max_df,
        filter_boilerplate=filter_bp,
        lexicon=lex,
        binary_meanings=args.binary_meanings,
    )

    col_names = [f"meaning_{i+1:02d}" for i in range(len(top_terms))]
    for i, name in enumerate(col_names):
        df[name] = X_meaning[:, i]

    n_rows = X_meaning.shape[0]
    n_all_meaning_zero = int(np.sum(X_meaning.sum(axis=1) == 0))

    if args.add_meta_columns:
        if "pub_date" not in df.columns:
            raise ValueError("--add-meta-columns 는 'pub_date' 열이 필요합니다.")
        df[TIMESTAMP_COL] = pd.to_datetime(df["pub_date"], errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        df[ANOMALY_ID_COL] = np.arange(len(df), dtype=np.int64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")

    payload = {
        "meaning_columns": col_names,
        "phrases": top_terms,
        "column_to_phrase": dict(zip(col_names, top_terms, strict=True)),
        "phrase_doc_frequency": dict(zip(col_names, top_doc_freq, strict=True)),
        "phrase_rank_score": dict(zip(col_names, top_scores, strict=True)),
        "rank_score_is_doc_frequency": True,
        "filter_boilerplate": filter_bp,
        "lexicon_json": str(args.lexicon_json.resolve()) if filter_bp else None,
        "meaning_columns_encoding": "binary_0_1" if args.binary_meanings else "count_per_summary",
        "count_vectorizer": {
            "ngram_range": [args.ngram_min, args.ngram_max],
            "min_df": args.min_df,
            "max_df": args.max_df,
            "vocabulary_size_fitted": len(vec.vocabulary_),
        },
        "source_csv": str(args.input.resolve()),
        "output_csv": str(args.output.resolve()),
        "rows": len(df),
        "rows_all_meaning_features_zero": n_all_meaning_zero,
    }
    args.vocab_json.parent.mkdir(parents=True, exist_ok=True)
    with args.vocab_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    n_docs = len(texts)
    pct_zero = 100.0 * float(n_all_meaning_zero) / max(n_rows, 1)
    print(f"Wrote {args.output} ({len(df)} rows, +{len(col_names)} meaning columns)")
    print(f"Wrote {args.vocab_json}")
    enc = "0/1 (존재 여부)" if args.binary_meanings else "등장 횟수 (0 이상 정수)"
    print(
        f"\nmeaning 열 값: {enc}. "
        f"전부 0인 행: {n_all_meaning_zero} / {n_rows} ({pct_zero:.2f}%) "
        f"(선택 {len(col_names)}개 n-gram이 해당 summary에 한 번도 없음)"
    )
    print(f"\n상위 구절 (df = 등장 문서 수 / 전체 {n_docs}건, 순위=df 내림차순):")
    for rank, (name, phrase, df_n) in enumerate(
        zip(col_names, top_terms, top_doc_freq, strict=True), start=1
    ):
        pct = 100.0 * float(df_n) / max(n_docs, 1)
        print(f"  {rank:2d}. {name}  df={df_n:6d} ({pct:5.1f}%)  {phrase!r}")


if __name__ == "__main__":
    main()

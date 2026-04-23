#!/usr/bin/env python
"""
Detect likely movie franchises from IMDb-style title data.

Approach:
1. Build title features (normalized text, tokens, sequel markers, metadata).
2. Generate candidate movie pairs with lightweight blocking.
3. Score pairs and link high-confidence matches.
4. Build connected components as franchise groups.
5. Number movies by release year inside each franchise.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "without",
    "und",
    "der",
    "die",
    "das",
    "de",
    "des",
    "du",
    "la",
    "le",
    "les",
    "el",
    "los",
    "las",
    "y",
}

SEQUEL_WORDS = {
    "part",
    "chapter",
    "episode",
    "volume",
    "vol",
    "tome",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
    "sequel",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
SEQUEL_TITLE_PATTERN = re.compile(
    r"(?:\b(part|chapter|episode|volume|vol|tome)\b\s*(\d+|[ivx]+)\b|"
    r"\b(ii|iii|iv|v|vi|vii|viii|ix|x)\b$|"
    r"\b#\s*([2-9][0-9]?)\b|"
    r"\b([2-9][0-9]?)\b$)",
    flags=re.IGNORECASE,
)
SEQUEL_KEYWORD_PATTERN = re.compile(
    r"\b(sequel|part two|part 2|part ii|second part|continuation|follow up|follow-up)\b",
    flags=re.IGNORECASE,
)
DIRECTOR_ID_PATTERN = re.compile(r"(nm\d+)")


@dataclass
class PairMeta:
    score: float
    shared_tokens: int
    jaccard: float
    same_root: bool
    same_prefix2: bool
    same_director: bool
    year_gap: Optional[float]
    any_marker: bool


class UnionFind:
    def __init__(self, items: Iterable[int]) -> None:
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: int) -> int:
        parent = self.parent[x]
        if parent != x:
            self.parent[x] = self.find(parent)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect likely franchises from Movie_50k.csv")
    parser.add_argument("--input", default="Data/Movie_50k.csv", help="Input CSV path")
    parser.add_argument(
        "--output",
        default="Data/franchise_movies_indexed.csv",
        help="Output CSV with movie_name, franchise_number, franchise_id",
    )
    parser.add_argument(
        "--debug-output",
        default="Data/franchise_movies_indexed_debug.csv",
        help="Debug CSV with extra matching fields",
    )
    parser.add_argument(
        "--stats-output",
        default="Data/franchise_group_stats.csv",
        help="Franchise-level stats CSV",
    )
    parser.add_argument("--min-score", type=float, default=4.5, help="Base score threshold for linking a pair")
    parser.add_argument(
        "--max-root-block",
        type=int,
        default=180,
        help="Max rows allowed in one root block before skipping pair expansion",
    )
    parser.add_argument(
        "--max-prefix-block",
        type=int,
        default=150,
        help="Max rows allowed in one prefix2 block before skipping pair expansion",
    )
    parser.add_argument(
        "--max-sig-block",
        type=int,
        default=140,
        help="Max rows allowed in one signature-token block before skipping pair expansion",
    )
    return parser.parse_args()


def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def normalize_title(text: str) -> str:
    t = strip_accents(str(text or "")).lower()
    t = re.sub(r"\([^)]*\)", " ", t)
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9:# ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def roman_to_int(token: str) -> Optional[int]:
    token = token.lower().strip()
    if not token:
        return None
    values = {"i": 1, "v": 5, "x": 10}
    if any(ch not in values for ch in token):
        return None
    total = 0
    prev = 0
    for ch in reversed(token):
        val = values[ch]
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total if total >= 1 else None


def extract_sequel_number(title_norm: str) -> Optional[int]:
    match = SEQUEL_TITLE_PATTERN.search(title_norm)
    if not match:
        return None
    parts = [p for p in match.groups() if p]
    if not parts:
        return None
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
        value = roman_to_int(part)
        if value is not None:
            return value
    return None


def has_sequel_marker(title_norm: str, keywords: str) -> bool:
    kw = str(keywords or "")
    return bool(SEQUEL_TITLE_PATTERN.search(title_norm) or SEQUEL_KEYWORD_PATTERN.search(kw))


def extract_root_key(title_norm: str) -> str:
    t = title_norm
    if ":" in t:
        t = t.split(":", 1)[0].strip()
    t = re.sub(r"\b(part|chapter|episode|volume|vol|tome)\b\s*(\d+|[ivx]+)\b", "", t)
    t = re.sub(r"\b(ii|iii|iv|v|vi|vii|viii|ix|x)\b$", "", t).strip()
    t = re.sub(r"\b#\s*([2-9][0-9]?)\b$", "", t).strip()
    t = re.sub(r"\b([2-9][0-9]?)\b$", "", t).strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize_informative(title_norm: str) -> List[str]:
    tokens = []
    for token in TOKEN_PATTERN.findall(title_norm):
        if len(token) <= 1:
            continue
        if token in STOPWORDS:
            continue
        if token in SEQUEL_WORDS:
            continue
        tokens.append(token)
    return tokens


def parse_first_director_id(value: str) -> str:
    text = str(value or "")
    match = DIRECTOR_ID_PATTERN.search(text)
    return match.group(1) if match else ""


def choose_movie_name(df: pd.DataFrame) -> pd.Series:
    english = df.get("englishTitle", "").fillna("").astype(str).str.strip()
    original = df.get("originalTitle", "").fillna("").astype(str).str.strip()
    return english.where(~english.eq(""), original)


def is_adult(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["releaseYear"] = pd.to_numeric(work.get("releaseYear"), errors="coerce")
    work = work.loc[~work.get("isAdult", "").map(is_adult)].copy()
    work = work.loc[work["releaseYear"].notna()].copy()
    work["movie_name"] = choose_movie_name(work)
    work = work.loc[work["movie_name"].str.len() > 0].copy()

    work["title_norm"] = work["movie_name"].map(normalize_title)
    work["root_key"] = work["title_norm"].map(extract_root_key)
    work["sequel_number"] = work["title_norm"].map(extract_sequel_number)
    work["sequel_marker"] = [
        has_sequel_marker(t, k)
        for t, k in zip(work["title_norm"], work.get("keywords", pd.Series("", index=work.index)))
    ]
    work["director_id"] = work.get("directors", "").map(parse_first_director_id)

    tokens = work["title_norm"].map(tokenize_informative)
    work["tokens"] = tokens
    work["token_set"] = tokens.map(frozenset)
    work["prefix2"] = tokens.map(lambda xs: tuple(xs[:2]) if len(xs) >= 2 else tuple(xs))

    has_text_signal = (work["root_key"].str.len().fillna(0) >= 3) | (work["token_set"].map(len) >= 2)
    work = work.loc[has_text_signal].copy()
    return work


def compute_token_idf(token_sets: Sequence[Set[str]]) -> Dict[str, float]:
    dfreq: Counter[str] = Counter()
    n_docs = len(token_sets)
    for s in token_sets:
        for token in s:
            dfreq[token] += 1
    idf = {}
    for token, freq in dfreq.items():
        idf[token] = math.log((n_docs + 1) / (freq + 1)) + 1.0
    return idf


def signature_tokens(tokens: Sequence[str], idf: Dict[str, float], top_k: int = 4) -> Tuple[str, ...]:
    if not tokens:
        return tuple()
    uniq = sorted(set(tokens), key=lambda t: (-idf.get(t, 0.0), t))
    return tuple(uniq[:top_k])


def generate_candidate_pairs(
    feat: pd.DataFrame,
    max_root_block: int,
    max_prefix_block: int,
    max_sig_block: int,
) -> Set[Tuple[int, int]]:
    blocks: Dict[str, List[int]] = defaultdict(list)

    for row in feat.itertuples():
        idx = int(row.Index)

        root = str(row.root_key or "")
        if len(root) >= 3:
            blocks[f"root:{root}"].append(idx)

        prefix2 = tuple(row.prefix2) if isinstance(row.prefix2, tuple) else tuple()
        if len(prefix2) == 2:
            blocks[f"p2:{prefix2[0]}|{prefix2[1]}"] .append(idx)

        sig = tuple(row.signature)
        for a, b in itertools.combinations(sig[:3], 2):
            blocks[f"sig2:{a}|{b}"].append(idx)

    pairs: Set[Tuple[int, int]] = set()
    for key, items in blocks.items():
        if len(items) < 2:
            continue

        if key.startswith("root:") and len(items) > max_root_block:
            continue
        if key.startswith("p2:") and len(items) > max_prefix_block:
            continue
        if key.startswith("sig2:") and len(items) > max_sig_block:
            continue

        items = sorted(set(items))
        for a, b in itertools.combinations(items, 2):
            pairs.add((a, b))
    return pairs


def score_pair(a: pd.Series, b: pd.Series) -> PairMeta:
    tokens_a = set(a["token_set"])
    tokens_b = set(b["token_set"])
    shared = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b) or 1
    jaccard = shared / union

    same_root = bool(a["root_key"]) and a["root_key"] == b["root_key"]
    same_prefix2 = tuple(a["prefix2"]) == tuple(b["prefix2"]) and len(tuple(a["prefix2"])) == 2
    same_director = bool(a["director_id"]) and a["director_id"] == b["director_id"]
    any_marker = bool(a["sequel_marker"]) or bool(b["sequel_marker"])

    year_gap = None
    if pd.notna(a["releaseYear"]) and pd.notna(b["releaseYear"]):
        year_gap = abs(float(a["releaseYear"]) - float(b["releaseYear"]))

    score = 0.0
    if same_root:
        score += 2.5
    if shared >= 2:
        score += 2.0
    elif shared == 1:
        score += 0.5
    if jaccard >= 0.5:
        score += 1.5
    elif jaccard >= 0.3:
        score += 0.5
    if same_prefix2:
        score += 2.0
    if same_director:
        score += 1.0
    if any_marker and shared >= 1:
        score += 1.0
    if a["sequel_marker"] and b["sequel_marker"]:
        score += 0.5
    if year_gap is not None:
        if year_gap <= 15:
            score += 1.0
        elif year_gap <= 30:
            score += 0.5

    return PairMeta(
        score=score,
        shared_tokens=shared,
        jaccard=jaccard,
        same_root=same_root,
        same_prefix2=same_prefix2,
        same_director=same_director,
        year_gap=year_gap,
        any_marker=any_marker,
    )


def should_link(meta: PairMeta, min_score: float) -> bool:
    if meta.score >= min_score:
        return True
    if meta.score >= 4.0 and (meta.same_root or meta.same_prefix2) and meta.shared_tokens >= 1:
        return True
    if meta.score >= 3.8 and meta.shared_tokens >= 2 and (meta.any_marker or meta.same_prefix2):
        return True
    return False


def component_is_valid(component_rows: pd.DataFrame, token_dfreq: Counter[str], n_titles: int) -> bool:
    if len(component_rows) < 2:
        return False

    n_movies = len(component_rows)
    n_marked = int(component_rows["sequel_marker"].sum())
    year_span = float(component_rows["releaseYear"].max() - component_rows["releaseYear"].min())

    token_sets = [set(x) for x in component_rows["token_set"] if x]
    if not token_sets:
        return False
    common = set.intersection(*token_sets) if token_sets else set()

    prefix2_values = [tuple(x) for x in component_rows["prefix2"] if isinstance(x, tuple) and len(tuple(x)) == 2]
    prefix2_counts = Counter(prefix2_values)
    dominant_prefix2_count = 0
    if prefix2_counts:
        dominant_prefix2_count = prefix2_counts.most_common(1)[0][1]

    if n_marked >= 1:
        if year_span > 40:
            return False
        if n_movies >= 12 and n_marked < 3:
            return False
        if n_movies >= 8 and n_marked < 2:
            return False
        if common:
            return True
        if dominant_prefix2_count >= max(2, n_movies - 1):
            return True
        return n_marked >= max(1, math.ceil(0.4 * n_movies))

    # No-marker components: keep only strict 2-title pairs with strong rare anchor.
    if n_movies != 2 or year_span > 15:
        return False
    if dominant_prefix2_count != 2:
        return False
    if not common:
        return False

    rare_limit = min(120, max(20, int(0.0025 * n_titles)))
    rare_common = [t for t in common if token_dfreq.get(t, 0) <= rare_limit]
    return len(rare_common) >= 2


def assign_franchise_ids(linked_rows: pd.DataFrame) -> pd.DataFrame:
    ordering = (
        linked_rows.groupby("component_id")
        .agg(first_year=("releaseYear", "min"), first_title=("movie_name", "min"))
        .sort_values(["first_year", "first_title", "component_id"])
        .reset_index()
    )
    ordering["franchise_id"] = range(1, len(ordering) + 1)
    out = linked_rows.merge(ordering[["component_id", "franchise_id"]], on="component_id", how="left")
    return out


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = pd.read_csv(input_path, low_memory=False)
    feat = build_feature_frame(raw)
    feat = feat.reset_index(drop=True)
    feat["row_id"] = feat.index

    token_sets = [set(x) for x in feat["token_set"]]
    token_idf = compute_token_idf(token_sets)
    token_dfreq = Counter()
    for s in token_sets:
        token_dfreq.update(s)
    n_titles = len(feat)

    feat["signature"] = feat["tokens"].map(lambda xs: signature_tokens(xs, token_idf, top_k=4))

    print(f"Input rows: {len(raw):,}")
    print(f"Feature rows used for matching: {len(feat):,}")

    candidates = generate_candidate_pairs(
        feat=feat,
        max_root_block=args.max_root_block,
        max_prefix_block=args.max_prefix_block,
        max_sig_block=args.max_sig_block,
    )
    print(f"Candidate pairs: {len(candidates):,}")

    uf = UnionFind(feat["row_id"].tolist())
    linked_meta: List[Tuple[int, int, PairMeta]] = []

    feat_idx = feat.set_index("row_id")
    for a_id, b_id in candidates:
        a = feat_idx.loc[a_id]
        b = feat_idx.loc[b_id]
        meta = score_pair(a, b)
        if should_link(meta, min_score=args.min_score):
            uf.union(a_id, b_id)
            linked_meta.append((a_id, b_id, meta))

    print(f"Linked pairs: {len(linked_meta):,}")

    feat["component_id"] = feat["row_id"].map(uf.find)
    comp_sizes = feat["component_id"].value_counts()
    candidate_components = comp_sizes[comp_sizes >= 2].index.tolist()

    valid_components: List[int] = []
    for cid in candidate_components:
        comp_rows = feat.loc[feat["component_id"] == cid]
        if component_is_valid(comp_rows, token_dfreq=token_dfreq, n_titles=n_titles):
            valid_components.append(cid)

    franchised = feat.loc[feat["component_id"].isin(valid_components)].copy()
    print(f"Detected franchise components: {len(valid_components):,}")
    print(f"Movies in detected components: {len(franchised):,}")

    if franchised.empty:
        raise RuntimeError("No franchise components detected. Lower thresholds and rerun.")

    franchised = assign_franchise_ids(franchised)

    franchised["releaseYearSort"] = franchised["releaseYear"].fillna(9999)
    franchised["sequelNumSort"] = franchised["sequel_number"].fillna(999)
    franchised = franchised.sort_values(
        ["franchise_id", "releaseYearSort", "sequelNumSort", "movie_name", "titleId"]
    ).copy()
    franchised["franchise_number"] = franchised.groupby("franchise_id").cumcount() + 1

    main_out = franchised[["movie_name", "franchise_number", "franchise_id"]].copy()
    debug_out = franchised[
        [
            "titleId",
            "movie_name",
            "releaseYear",
            "franchise_id",
            "franchise_number",
            "root_key",
            "sequel_marker",
            "sequel_number",
            "director_id",
            "tokens",
        ]
    ].copy()
    stats_out = (
        franchised.groupby("franchise_id")
        .agg(
            n_movies=("titleId", "count"),
            first_year=("releaseYear", "min"),
            last_year=("releaseYear", "max"),
            n_marked=("sequel_marker", "sum"),
            representative_title=("movie_name", "min"),
        )
        .reset_index()
        .sort_values(["n_movies", "first_year"], ascending=[False, True])
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    main_out.to_csv(output_path, index=False)
    Path(args.debug_output).parent.mkdir(parents=True, exist_ok=True)
    debug_out.to_csv(args.debug_output, index=False)
    Path(args.stats_output).parent.mkdir(parents=True, exist_ok=True)
    stats_out.to_csv(args.stats_output, index=False)

    hp_hits = debug_out.loc[debug_out["movie_name"].str.contains("harry potter", case=False, na=False)]
    print(f"Wrote: {output_path}")
    print(f"Wrote: {args.debug_output}")
    print(f"Wrote: {args.stats_output}")
    print(f"Harry Potter rows in detected franchises: {len(hp_hits):,}")


if __name__ == "__main__":
    main()

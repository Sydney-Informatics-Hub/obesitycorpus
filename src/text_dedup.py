from nltk.util import ngrams
#from likeable.structure import minhash
from datasketch import MinHash, LeanMinHash
import itertools
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
import sys
import xxhash

def minhash(string_set):
    hashers = [xxhash.xxh32(w.encode("utf8")) for w in string_set]
    while True:
        yield min(h.intdigest() for h in hashers)
    for h in hashers:
        h.update(".")
    return  hashers

class Corpus:
    def __init__(self, make_ngrams, n, n_hashes=30):
        self.make_ngrams = make_ngrams
        self.hash_buckets = [defaultdict(list) for i in range(n_hashes)]
        for i in tqdm(range(n), desc="Hashing"):
            doc_ngrams = make_ngrams(i)
            for h, b in zip(minhash(doc_ngrams), self.hash_buckets):
                b[h].append(i)

    def find_similar(self, other, max_hash_distance=0.2, check_inclusion=None):
        # TODO: compute in chunks, or just use bin check
        hits = Counter()
        for b_idx, bucket in enumerate(tqdm(self.hash_buckets, desc="Finding hits")):
            for h, b in bucket.items():
                other_b = other.hash_buckets[b_idx].get(h, [])
                vals = itertools.product(b, other_b)
                if self is other:
                    vals = ((x, y) for x, y in vals if x != y)
                hits.update(vals)

        threshold = (1 - max_hash_distance) * len(self.hash_buckets)
        for (self_idx, other_idx), n in tqdm(hits.items(), desc="Processing hits"):
            if n < threshold:
                continue
            self_ngrams = set(self.make_ngrams(self_idx))
            other_ngrams = set(other.make_ngrams(other_idx))
            yield self_idx, other_idx, len(self_ngrams & other_ngrams) / len(
                self_ngrams | other_ngrams
            )


def deduplicate(
    df,
    *,
    max_jaccard,
    max_hash_distance=0.3,
    ngram_size=3,
    first_fetch_col="first_fetch",
    last_fetch_col="last_fetch",
    groupby_col=None,
    text_cols=("headline", "body_text"),
    id_col="article_id",
    n_hashes=30,
):
    def _mark_duplicates(df):
        def make_ngrams(idx, truncate=1500):
            return [
                " ".join(filter(None, gram))
                for gram in ngrams(
                    (
                        df.iloc[idx].loc[list(text_cols)].fillna("").str.join("\n")
                    ).split()[:truncate],
                    ngram_size,
                    pad_left=True,
                )
            ]

        def _within_1d(idx1, idx2):
            # Currently unused
            return abs(
                df[first_fetch_col].iloc[idx1] - df[last_fetch_col].iloc[idx2]
            ) <= pd.to_timedelta("1d")

        out = pd.Series(-1, index=df.index, dtype=int)
        corpus = Corpus(make_ngrams, len(df), n_hashes=n_hashes)
        for idx1, idx2, sim in corpus.find_similar(
            corpus, max_hash_distance=max_hash_distance
        ):
            if sim > max_jaccard:
                if df.iloc[idx1][first_fetch_col] > df.iloc[idx2][first_fetch_col]:
                    idx1, idx2 = idx2, idx1
                orig_id = out.iloc[idx1]
                if orig_id == -1:
                    orig_id = df.iloc[idx1][id_col]
                out.iloc[idx2] = orig_id
        df["duplicate_of"] = out
        print(
            f"Found {(out > -1).sum()} duplicates of {len(df)} "
            f"in {df.iloc[0][groupby_col] if groupby_col else 'frame'}",
            file=sys.stderr,
        )
        return out

    text_cols = list(text_cols)

    if groupby_col:
        duplicate_of = df.groupby(groupby_col, as_index=False).apply(_mark_duplicates)
    else:
        duplicate_of = _mark_duplicates(df)
    # drop the grouping index
    duplicate_of.index = duplicate_of.index.get_level_values(-1)
    df["duplicate_of"] = duplicate_of
    df = df.set_index(id_col)
    mask = df.duplicate_of >= 0
    print(f"Found {mask.sum()} duplicates", file=sys.stderr)
    for duplicate_of, samples in df[mask].groupby("duplicate_of"):
        df.loc[duplicate_of, last_fetch_col] = max(
            df.loc[duplicate_of, last_fetch_col], samples[last_fetch_col].max()
        )
    return df[~mask].reset_index().drop("duplicate_of", axis=1)



import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--in-csv", type=argparse.FileType('r'), default=sys.stdin)
ap.add_argument("-o", "--out-csv", type=argparse.FileType('w'), default=sys.stdout)
ap.add_argument("-g", "--group-by", default=None)
ap.add_argument(
    "--text-cols",
    type=lambda s: s.split(","),
    default=["title", "body"],
    help="Comma-delimited list of text fields.",
)
ap.add_argument("--id-col", default="article_id")
ap.add_argument("--ngram-size", type=int, default=3)
ap.add_argument("--max-jaccard", type=float, default=0.9)
ap.add_argument("--max-hash-distance", type=float, default=0.2)
ap.add_argument("--n-hashes", type=int, default=30)
args = ap.parse_args()
df = pd.read_csv(args.in_csv)
df = deduplicate(
    df,
    id_col=args.id_col,
    text_cols=args.text_cols,
    groupby_col=args.group_by,
    ngram_size=args.ngram_size,
    max_jaccard=args.max_jaccard,
    max_hash_distance=args.max_hash_distance,
    n_hashes=args.n_hashes,
)
df.to_csv(args.out_csv, index=False)

# %%
import pandas as pd
from utils import get_project_root
from datasketch import MinHash, LeanMinHash
import xxhash

import pathlib
from utils import get_projectpaths
(projectroot, rawdatapath, cleandatapath, processeddatapath) = get_projectpaths()

# %% corpusdf must have title, body and source columns
corpusdf = pd.read_pickle(cleandatapath/"corpusdf.pickle")

# %% test
testbodies = corpusdf.body.head(20).to_list()

def minhash(seq, num_perm):
    m = MinHash(num_perm=num_perm, hashfunc=xxhash.xxh64_intdigest)
    for s in seq:
        m.update(s.encode('utf8'))
    return LeanMinHash(m)



minhashes = []
for i in range(100):
    minhashes.append(minhash(tokenize(testbodies[i])))

mh_distances = {}
for i in range(100):
    for j in range(100):
        if i < j:
            mh_distances[(i, j)] = minhashes[i].jaccard(minhashes[j])
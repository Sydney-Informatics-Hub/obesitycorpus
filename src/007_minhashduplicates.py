# %%
import pandas as pd
from utils import get_project_root
from datasketch import MinHash, MinHashLSH
import xxhash
from gensim.utils import tokenize
import pathlib
from utils import get_projectpaths
(projectroot, rawdatapath, cleandatapath, processeddatapath) = get_projectpaths()
import seaborn as sns
from diffviz import html_diffs
import collections
import nltk
from functs import get_record_by_article_id, expand_source
from itertools import chain

# %% CLIENT SPECIFIES THESE VARIABLES
similarity_cutoff = 0.7
# anything with >= the below cutoff will be dropped from the same source
dropping_similarity_cutoff = 1
num_perm = 256

# %% FUNCTION DEFINITION BLOCK ------------------------------------------------



def get_hash_df(df,  num_perm=128):
    """
    Returns a dictionary of {(article_id1, article_id2): dis}
    """
    n_files = df.shape[0]
    print(n_files)
    minhashes = []
    for i in range(n_files):
        # number of different hash functions to use
        # setting it higher improves accuracy but runs slower
        minhashes.append(minhash(tokenize(df.body[i]), num_perm=num_perm))
    mh_distances = {}
    for i in range(n_files):
        for j in range(n_files):
            if i < j:
                mh_distances[(df.article_id[i], df.article_id[j])] = minhashes[i].jaccard(minhashes[j])
    minhashdf = pd.DataFrame.from_dict(mh_distances, orient='index', columns=['similarity_score'])
    minhashdf[['article_id', 'compared_article_id']] = pd.DataFrame(minhashdf.index.tolist(), index=minhashdf.index)
    return minhashdf

def cbind_with_tophashes(corpusdf, minhashdf):
    # NB this will result in duplicate lines for some articles
    # where the minhash is the same for more than one article
    minhash_onlytop = (minhashdf[
        minhashdf.groupby(['article_id'])
        ['similarity_score']
        .transform(max) 
        # and return True for the position where that max equals similarity score
        == minhashdf['similarity_score']
        ])

    # the below df copy is needed to make sure if article_id_later is best
    # matched by an earlier id we report that one as the best match
    minhash_onlytop_dupe = minhash_onlytop.copy()
    minhash_onlytop_dupe = (minhash_onlytop_dupe.rename(
        columns={
            "article_id": "compared_article_id", 
            "compared_article_id": "article_id"}))
    minhash_onlytop_formerge = minhash_onlytop.append(minhash_onlytop_dupe, ignore_index=True)

    minhash_onlytop_formerge = (minhash_onlytop_formerge[
        minhash_onlytop_formerge.groupby(['article_id'])
        ['similarity_score']
        .transform(max) 
        # and return True for the position where that max equals similarity score
        == minhash_onlytop_formerge['similarity_score']
        ])
    corpusdf_with_mostsimilar = corpusdf.merge(minhash_onlytop_formerge, on='article_id', how='inner')
    return corpusdf_with_mostsimilar

def get_smaller_id(article_id, compared_article_id):
    '''
    ['a', 'b', 'b','c', 'c', 'd', 'd', 'e', 'f'] and
    ['e', 'c', 'd','b', 'd', 'c', 'b', 'a', 'z']
    Returns 
    {'a': ['e'], 'b': ['c', 'd'], 'f': ['z']}
    '''
    mydict = {}
    already_considered = []
    for i in range(len(article_id)):
        if (article_id[i] in already_considered):
            # the article is already a value in the dict
            pass
        elif article_id[i] in mydict:
            # already in dict as key
            mydict[article_id[i]] = mydict[article_id[i]] + [compared_article_id[i]]
            already_considered.append(compared_article_id[i])        
        else:
            # this is a new text id we haven't seen before
            mydict[article_id[i]] = [compared_article_id[i]]
            already_considered.append(compared_article_id[i]) 
    return mydict

def get_unique_df_and_hashes(corpusdf, source, num_perm):
    if source != None:
        corpusdf = corpusdf[corpusdf.source == source].reset_index(drop=True)
    minhashdf = get_hash_df(corpusdf, num_perm=128)

    corpusdf_with_mostsimilar = cbind_with_tophashes(corpusdf=corpusdf, minhashdf=minhashdf).reset_index(drop=True)
    # pull out identicals
    corpusdf_with_mostsimilar_identicals = corpusdf_with_mostsimilar[corpusdf_with_mostsimilar.similarity_score == 1]
    corpusdf_with_mostsimilar_identicals.to_csv(processeddatapath/"corpusdf_with_mostsimilar_identicals.csv", index=False)

    #
    mydf = minhashdf[minhashdf.similarity_score == 1]
    identicals_dict = get_smaller_id(mydf.article_id.tolist(), mydf.compared_article_id.tolist())

    corpusdf_with_mostsimilar_identicals_keep = corpusdf_with_mostsimilar_identicals[corpusdf_with_mostsimilar_identicals.article_id.isin(identicals_dict.keys())]
    dropped = corpusdf_with_mostsimilar_identicals[~(corpusdf_with_mostsimilar_identicals.article_id.isin(identicals_dict.keys()))]

    # get non-identicals 
    corpusdf_with_mostsimilar_nonidenticals = corpusdf_with_mostsimilar[~(corpusdf_with_mostsimilar.similarity_score == 1)]

    deduped_corpusdf = corpusdf_with_mostsimilar_nonidenticals.append(corpusdf_with_mostsimilar_identicals_keep)

    return (minhashdf, identicals_dict, corpusdf_with_mostsimilar, dropped, deduped_corpusdf)

def merge_dicts(*dicts):
    """
    merge_dicts(
    {'a': ['e'], 'b': ['c', 'd'], 'f': ['z']},
    {'a': ['n', 'm'], 'f': ['um']})
    # returns:
    defaultdict(list, {'a': ['e', 'n', 'm'], 'b': ['c', 'd'], 'f': ['z', 'um']})
    """
    merged = collections.defaultdict(list)
    
    for d in dicts:
        for k, v in d.items():
            merged[k] = merged[k] + v
    return merged


# def gather_deduped_corpus_bysource(corpusdf, num_perm):
#     for idx, source in enumerate(corpusdf.source.unique().tolist()):
#         if idx == 0:
#             minhashdf, identicals_dict, corpusdf_with_mostsimilar, dropped, deduped_corpusdf = get_unique_df_and_hashes(corpusdf, source=source, num_perm=num_perm)
#         else:
#             minhashdf2, identicals_dict2, corpusdf_with_mostsimilar2, dropped2, deduped_corpusdf2 = get_unique_df_and_hashes(corpusdf, source=source, num_perm=num_perm)
#             minhashdf = minhashdf.append(minhashdf2).reset_index(drop=True)
#             corpusdf_with_mostsimilar = corpusdf_with_mostsimilar.append(corpusdf_with_mostsimilar2).reset_index(drop=True)
#             dropped = dropped.append(dropped2).reset_index(drop=True)
#             deduped_corpusdf = deduped_corpusdf.append(deduped_corpusdf2).reset_index(drop=True)
#             identicals_dict = merge_dicts(identicals_dict, identicals_dict2)
#     return (minhashdf, identicals_dict, corpusdf_with_mostsimilar, dropped, deduped_corpusdf)



# %% LOAD DATA BLOCK ---------------------------------------------------------
# corpusdf must have title, body and source columns
corpusdf = pd.read_pickle(processeddatapath/'corpusdf_with_wc.pickle')
# WORK BLOCK --------------------------------------------------------------

# %%

def make_text_hash(text, num_perm=num_perm):
    myset = set(tokenize(text))
    hash1 = MinHash(num_perm=num_perm)
    for d in myset:
        hash1.update(d.encode('utf8'))
    return hash1

corpusdf['hash'] = corpusdf.apply(lambda x: make_text_hash(x.body), axis=1) 
#corpusdf.to_pickle(processeddatapath/'corpusdf_with_wc_hash.pickle')

# %%

# Create LSH index
lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)

for index, row in corpusdf.iterrows():
    lsh.insert(row['article_id'], row['hash'])


def get_matches(hash, article_id):
    matches = lsh.query(hash)
    matches.remove(article_id)
    return matches

corpusdf['matched_list'] = corpusdf.apply(
    lambda x: get_matches(x.hash, x.article_id), 
    axis=1) 



# %%
def my_jaccard(set1, set2):
    if len(set1.union(set2)) != 0:
        return len(set1.intersection(set2)) / len(set1.union(set2))
    else:
        # the sets have nothing in common
        # to avoid divide by 0 error
        return 0

def get_jaccards(corpusdf, original, matched_list, ngram_size):
    body1 = corpusdf[(corpusdf['article_id'] == original)]['body'].values[0].lower()
    set1 = set(nltk.ngrams(tokenize(body1), n=ngram_size))
    jaccards = []
    # no matches for this article
    if len(matched_list) == 0:
        return []
    else:
        for id in matched_list:
           body2 = corpusdf[(corpusdf['article_id'] == id)]['body'].values[0].lower()
           set2 = set(nltk.ngrams(tokenize(body2), n=ngram_size))
           jaccard = my_jaccard(set1, set2)
           jaccards.append(jaccard)
        return jaccards


corpusdf['jaccards'] = corpusdf.apply(lambda x: get_jaccards(corpusdf=corpusdf, original=x.article_id, matched_list= x.matched_list, ngram_size=1), axis=1) 


# %%

intermediate_df = corpusdf[['matched_list', 'article_id', 'jaccards']].copy()
intermediate_df['listlen'] = intermediate_df.apply(lambda x: len(x.matched_list), axis = 1)
intermediate_df = intermediate_df[intermediate_df.listlen > 0]
intermediate_df['article_id_duped'] = intermediate_df.apply(lambda x: [x.article_id] * x.listlen, axis = 1)


def explode_list(df, col):
    return list(chain.from_iterable(df[col].to_list()))

deduplication_df = pd.DataFrame(
    {
        'docid2' :explode_list(intermediate_df, 'matched_list'),
        'similarity': explode_list(intermediate_df, 'jaccards'),
        'docid1':explode_list(intermediate_df, 'article_id_duped')
    }
)
deduplication_df['source1'] = expand_source(deduplication_df.apply(lambda x: x.docid1[0:2], axis=1))
deduplication_df['source2'] = expand_source(deduplication_df.apply(lambda x: x.docid2[0:2], axis=1))



# %% now we can plot
def plot_hash_similarity_by_source(df, source):
    # %% Visualise similarity scores
    if source is not None:
        df = df[df.source1 == source]
        df = df[df.source2 == source]
    else:
        source = "Entire corpus"

    plot = sns.histplot(data=(df[
        # return single row for article_id and similarity_score,
        # so one row per article for this plot    
        ~df[
        ['docid1',"similarity"]]
        .duplicated()]) , x="similarity").set_title(source)
    return plot


plot_hash_similarity_by_source(deduplication_df, source=None);


# %%

def get_duplicate_ids(df, sourcematch, similaritycutoff):
    '''
    sampledata = pd.DataFrame({
    'docid1': ['a'] * 3 + ['b'] * 3 + ['c'] * 3 + ['d'] * 3, 
    'docid2': ['b', 'c', 'd', 'a', 'c', 'd', 'a', 'b', 'd', 'a', 'b', 'c'], 
    'source1': ['x'] * 6 + ['y'] * 6, 
    'source2': ['x', 'y', 'y'] * 2 + ['x', 'x', 'y'] * 2, 
    'similarity': [1] * 12} )
    drop = get_duplicate_ids(sampledata, sourcematch=True, scorecutoff=1)
    # should return ['a'] when sourcematch=False & ['a', 'c'] when sourcematch=True
    '''
    df = df[df.similarity >= similaritycutoff]
    if sourcematch:
        df = df[df.source1 == df.source2]
    list1 = list(df['docid1'].values)
    list2 = list(df['docid2'].values)
    assert len(list1) == len(list2)
    considered, drop = ([] for i in range(2))
    for i in range(len(list1)):
        if list1[i] not in considered:
            considered.append(list1[i])
            considered.append(list2[i])
            # keep.append(list1[i])
            drop.append(list2[i])
        else:
            if list2[i] not in considered:
                considered.append(list2[i])
                drop.append(list2[i])
    drop = sorted(list(set(drop)))
    return drop



drop_source = get_duplicate_ids(deduplication_df, sourcematch=True, similaritycutoff=dropping_similarity_cutoff)
drop_all = get_duplicate_ids(deduplication_df, sourcematch=False, similaritycutoff=dropping_similarity_cutoff)


# %% get different subcorpora
corpusdf_deduped_by_source = corpusdf[~corpusdf.article_id.isin(drop_source)]
duplicates_dropped_by_source = corpusdf[corpusdf.article_id.isin(drop_source)]
corpus_df_uniq = corpusdf[~corpusdf.article_id.isin(drop_all)]
duplicates_dropped_entirecorpus = corpusdf[corpusdf.article_id.isin(drop_all)]

# %% 

def write_article_diffs(corpusdf, article_kept_id,article_dropped_id, jaccard, outdir):
    x = corpusdf[corpusdf.article_id == article_dropped_id].squeeze()
    y = corpusdf[corpusdf.article_id == article_kept_id].squeeze()
    title_a = f'Title: {x.title}'
    title_b = f'Title: {y.title}'
    metadata_a = f'{x.article_id} Jaccard:{jaccard} {x.source} filename: {x.year}-{x.original_numeric_month}-{x.fourdigitcode} metadata: {x.date.date()}'
    metadata_b = f'{y.article_id} Jaccard:{jaccard} {y.source} filename: {y.year}-{y.original_numeric_month}-{y.fourdigitcode} metadata: {y.date.date()}'
    with open(f'{outdir}/{article_dropped_id}.html', "w") as f:
        myhtml = html_diffs(x.body, y.body, title_a, title_b, metadata_a, metadata_b)
        f.write(myhtml)

# %% for corpus by source
for_diffs_dropped_bysource_df = deduplication_df[
    (deduplication_df.source1==deduplication_df.source2) & 
    deduplication_df.docid1.isin(drop_source) &
    (deduplication_df.similarity >= dropping_similarity_cutoff)]

# mkdir if doesn't exist
pathlib.Path(processeddatapath/"dropped_same_source").mkdir(parents=True, exist_ok=True)

for_diffs_dropped_bysource_df.apply(lambda x: write_article_diffs(
    corpusdf=corpusdf, 
    article_kept_id = x.docid1, 
    article_dropped_id = x.docid2, 
    outdir=(processeddatapath/"dropped_same_source")), axis=1);

# %% entire corpus - what's dropped to fully de-dupe
for_diffs_dropped_corpus_df = deduplication_df[ 
    deduplication_df.docid1.isin(drop_all) &
    (deduplication_df.similarity >= dropping_similarity_cutoff)]

# mkdir if doesn't exist
pathlib.Path(processeddatapath/"dropped_dedupe_corpus").mkdir(parents=True, exist_ok=True)

for_diffs_dropped_corpus_df.apply(lambda x: write_article_diffs(
    corpusdf=corpusdf, 
    article_kept_id = x.docid1, 
    article_dropped_id = x.docid2, 
    outdir=(processeddatapath/"dropped_dedupe_corpus")), axis=1);

# %% write out comparisons within source for similar but NOT dropped articles

# get the similars:

similar_source = set(get_duplicate_ids(
    deduplication_df, 
    sourcematch=True, 
    similaritycutoff=similarity_cutoff)) - set(drop_source)


for_diffs_undropped_similar = deduplication_df[
    (deduplication_df.source1==deduplication_df.source2) & 
    (deduplication_df.similarity < dropping_similarity_cutoff) &
    (deduplication_df.similarity >=  similarity_cutoff) &
    (deduplication_df.docid1.isin(similar_source))] 

# mkdir if doesn't exist
pathlib.Path(processeddatapath/"similar_same_source").mkdir(parents=True, exist_ok=True)

for_diffs_undropped_similar.apply(lambda x: write_article_diffs(
    corpusdf=corpusdf, 
    article_kept_id = x.docid1, 
    article_dropped_id = x.docid2, 
    jaccard = round(x.similarity, 3),
    outdir = (processeddatapath/"similar_same_source")), axis=1);




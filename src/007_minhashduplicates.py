# %%
import pandas as pd
from utils import get_project_root
from datasketch import MinHash, LeanMinHash
import xxhash
from gensim.utils import tokenize
import pathlib
from utils import get_projectpaths
(projectroot, rawdatapath, cleandatapath, processeddatapath) = get_projectpaths()
import seaborn as sns
from diffviz import html_diffs
import collections

# %% FUNCTION DEFINITION BLOCK ------------------------------------------------
def minhash(seq, num_perm):
    m = MinHash(num_perm=num_perm, hashfunc=xxhash.xxh64_intdigest)
    for s in seq:
        m.update(s.encode('utf8'))
    return LeanMinHash(m)


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


def gather_deduped_corpus_bysource(corpusdf, num_perm):
    for idx, source in enumerate(corpusdf.source.unique().tolist()):
        if idx == 0:
            minhashdf, identicals_dict, corpusdf_with_mostsimilar, dropped, deduped_corpusdf = get_unique_df_and_hashes(corpusdf, source=source, num_perm=num_perm)
        else:
            minhashdf2, identicals_dict2, corpusdf_with_mostsimilar2, dropped2, deduped_corpusdf2 = get_unique_df_and_hashes(corpusdf, source=source, num_perm=num_perm)
            minhashdf = minhashdf.append(minhashdf2).reset_index(drop=True)
            corpusdf_with_mostsimilar = corpusdf_with_mostsimilar.append(corpusdf_with_mostsimilar2).reset_index(drop=True)
            dropped = dropped.append(dropped2).reset_index(drop=True)
            deduped_corpusdf = deduped_corpusdf.append(deduped_corpusdf2).reset_index(drop=True)
            identicals_dict = merge_dicts(identicals_dict, identicals_dict2)
    return (minhashdf, identicals_dict, corpusdf_with_mostsimilar, dropped, deduped_corpusdf)



# %% LOAD DATA BLOCK ---------------------------------------------------------
# corpusdf must have title, body and source columns
corpusdf = pd.read_pickle(processeddatapath/'corpusdf_with_wc.pickle')
# WORK BLOCK --------------------------------------------------------------


# %% UNCOMMENT TO REGENERATE
corpusdf_adv = corpusdf[corpusdf.source == 'CanTimes']
# todo expand to entire corpus later
minhashdf, identicals_dict, corpusdf_with_mostsimilar, dropped, deduped_corpusdf = gather_deduped_corpus_bysource(corpusdf_adv, num_perm=128)
minhashdf.to_pickle(processeddatapath/"minhashdf.pickle")
corpusdf_with_mostsimilar.to_pickle(processeddatapath/"corpusdf_with_mostsimilar.pickle")
dropped.to_pickle(processeddatapath/"dropped.pickle")
deduped_corpusdf.to_pickle(processeddatapath/"deduped_corpusdf.pickle")


# %%
#minhashdf = pd.read_pickle(processeddatapath/"minhashdf.pickle")
corpusdf_with_mostsimilar = pd.read_pickle(processeddatapath/"corpusdf_with_mostsimilar.pickle")
dropped = pd.read_pickle(processeddatapath/"dropped.pickle")
deduped_corpusdf = pd.read_pickle(processeddatapath/"deduped_corpusdf.pickle")


# %% 
def plot_hash_similarity_by_source(corpusdf_with_mostsimilar, source):
    # %% Visualise similarity scores
    if source is not None:
        corpusdf_with_mostsimilar = corpusdf_with_mostsimilar[corpusdf_with_mostsimilar.source == source]
    else:
        source = "Entire corpus"

    plot = sns.histplot(data=(corpusdf_with_mostsimilar[
        # return single row for article_id and similarity_score,
        # so one row per article for this plot    
        ~corpusdf_with_mostsimilar[
        ['article_id',"similarity_score"]]
        .duplicated()]) , x="similarity_score").set_title(source)
    return plot
# %%
plot_hash_similarity_by_source(corpusdf_with_mostsimilar, source=None);




# TODO minhashdf.to_csv(processeddatapath/"minhash_distances.csv")


# todo corpusdf_with_mostsimilar write this to csv

# %% what's not quite identical?
corpusdf_with_mostsimilar_outliers = (corpusdf_with_mostsimilar[
    (corpusdf_with_mostsimilar.similarity_score < 1
    ) & (corpusdf_with_mostsimilar.similarity_score > 0.6)
    ])

# corpusdf_with_mostsimilar_outliers2 = (corpusdf_with_mostsimilar[
#     (corpusdf_with_mostsimilar.similarity_score < 0.6
#     ) & (corpusdf_with_mostsimilar.similarity_score > 0.4)
#     ])

# %%

def write_article_diffs(corpusdf, article_dropped_id,article_kept_id, outdir):
    x = corpusdf[corpusdf.article_id == article_dropped_id].squeeze()
    y = corpusdf[corpusdf.article_id == article_kept_id].squeeze()
    title_a = f'Title: {x.title}'
    title_b = f'Title: {y.title}'
    metadata_a = f'{x.article_id}: {x.source} filename: {x.year}-{x.original_numeric_month}-{x.fourdigitcode} metadata: {x.date.date()}'
    metadata_b = f'{y.article_id}: {y.source} filename: {y.year}-{y.original_numeric_month}-{y.fourdigitcode} metadata: {y.date.date()}'
    with open(f'{outdir}/{article_dropped_id}.html', "w") as f:
        myhtml = html_diffs(x.body, y.body, title_a, title_b, metadata_a, metadata_b)
        f.write(myhtml)


# %%
write_article_diffs(corpusdf, "AD091000016", "")




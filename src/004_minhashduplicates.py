# %%
import pandas as pd
from utils import get_project_root
from datasketch import MinHash, MinHashLSH
#import xxhash
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
import datetime as dt

# %% CLIENT SPECIFIES THESE VARIABLES
similarity_cutoff = 0.4
# anything with >= the below cutoff will be dropped from the same source
dropping_similarity_cutoff = 0.6
# anything with < ndays days between publication date from metadata will be dropped from the same source
ndays = 1
num_perm = 256

# %% FUNCTION DEFINITION BLOCK ------------------------------------------------
def make_text_hash(text, num_perm=num_perm):
    myset = set(tokenize(text))
    hash1 = MinHash(num_perm=num_perm)
    for d in myset:
        hash1.update(d.encode('utf8'))
    return hash1

def get_matches(hash, article_id):
    matches = lsh.query(hash)
    matches.remove(article_id)
    return matches

def find_jaccard(set1, set2):
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
           jaccard = find_jaccard(set1, set2)
           jaccards.append(jaccard)
        return jaccards

def explode_list(df, col):
    return list(chain.from_iterable(df[col].to_list()))

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

def get_duplicate_ids(df, sourcematch, similaritycutoff, ndays):
    '''
    sampledata = pd.DataFrame({
    'docid1': ['a'] * 3 + ['b'] * 3 + ['c'] * 3 + ['d'] * 3, 
    'docid2': ['b', 'c', 'd', 'a', 'c', 'd', 'a', 'b', 'd', 'a', 'b', 'c'], 
    'source1': ['x'] * 6 + ['y'] * 6, 
    'source2': ['x', 'y', 'y'] * 2 + ['x', 'x', 'y'] * 2, 
    'similarity': [1] * 12},
    'date1': ['2009-10-17'] * 12,
    'date2': ['2009-10-17'] * 12)
    drop = get_duplicate_ids(sampledata, sourcematch=True, scorecutoff=1)
    # should return ['a'] when sourcematch=False & ['a', 'c'] when sourcematch=True
    '''
    df = df[df.similarity >= similaritycutoff]
    df['timediff'] = abs(df.date1 - df.date2)
    df = df[df.timediff < dt.timedelta(days=ndays)]
    df = df[df.wordcount_total1 >= df.wordcount_total2]
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

def write_article_diffs(corpusdf, article_kept_id,article_dropped_id, jaccard, outdir):
    x = corpusdf[corpusdf.article_id == article_dropped_id].squeeze()
    y = corpusdf[corpusdf.article_id == article_kept_id].squeeze()
    title_a = f'Title dropped: {x.title}'
    title_b = f'Title kept: {y.title}'
    metadata_a = f'{x.article_id} Jaccard:{jaccard} {x.source} filename: {x.year}-{x.original_numeric_month}-{x.fourdigitcode} metadata: {x.date.date()}'
    metadata_b = f'{y.article_id} Jaccard:{jaccard} {y.source} filename: {y.year}-{y.original_numeric_month}-{y.fourdigitcode} metadata: {y.date.date()}'
    with open(f'{outdir}/{jaccard}_{article_dropped_id}.html', "w") as f:
        myhtml = html_diffs(x.body, y.body, title_a, title_b, metadata_a, metadata_b)
        f.write(myhtml)

# %% LOAD DATA BLOCK ---------------------------------------------------------
# corpusdf must have title, body and source columns
corpusdf = pd.read_pickle(processeddatapath/'corpusdf_with_wc.pickle')
# WORK BLOCK --------------------------------------------------------------

corpusdf['hash'] = corpusdf.apply(lambda x: make_text_hash(x.body), axis=1) 
#corpusdf.to_pickle(processeddatapath/'corpusdf_with_wc_hash.pickle')

# %%
# Create LSH index
lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)

for index, row in corpusdf.iterrows():
    lsh.insert(row['article_id'], row['hash'])

corpusdf['matched_list'] = corpusdf.apply(
    lambda x: get_matches(x.hash, x.article_id), 
    axis=1) 

corpusdf['jaccards'] = corpusdf.apply(
    lambda x: get_jaccards(
        corpusdf=corpusdf, 
        original=x.article_id, 
        matched_list= x.matched_list, 
        ngram_size=1), axis=1) 


# %%

intermediate_df = corpusdf[['matched_list', 'article_id', 'jaccards']].copy()
intermediate_df['listlen'] = intermediate_df.apply(lambda x: len(x.matched_list), axis = 1)
intermediate_df = intermediate_df[intermediate_df.listlen > 0]
intermediate_df['article_id_duped'] = intermediate_df.apply(lambda x: [x.article_id] * x.listlen, axis = 1)


deduplication_df = pd.DataFrame(
    {
        'docid2' :explode_list(intermediate_df, 'matched_list'),
        'similarity': explode_list(intermediate_df, 'jaccards'),
        'docid1':explode_list(intermediate_df, 'article_id_duped')
    }
)
deduplication_df['source1'] = expand_source(deduplication_df.apply(lambda x: x.docid1[0:2], axis=1))
deduplication_df['source2'] = expand_source(deduplication_df.apply(lambda x: x.docid2[0:2], axis=1))

# %% join with article dates (from metadata)
article_id_metadata_df = corpusdf[['article_id', 'date', 'wordcount_total']].copy()


deduplication_df = deduplication_df.merge(
    article_id_metadata_df, 
    left_on='docid1', 
    right_on='article_id', 
    how='inner').drop('article_id', axis=1).merge(
    article_id_metadata_df, 
    left_on='docid2', 
    right_on='article_id',
    suffixes=('1', '2'),
    how='left').drop('article_id', axis=1)

# %% now we can plot
plot_hash_similarity_by_source(deduplication_df, source=None);


# %%

drop_source = get_duplicate_ids(
    deduplication_df, 
    sourcematch=True, 
    similaritycutoff=dropping_similarity_cutoff,
    ndays=ndays)

similar_across_sources = get_duplicate_ids(
    deduplication_df, 
    sourcematch=False, 
    similaritycutoff=dropping_similarity_cutoff,
    ndays=ndays)

# %% get different subcorpora
corpusdf_deduped_by_source = corpusdf[~corpusdf.article_id.isin(drop_source)]
duplicates_dropped_by_source = corpusdf[corpusdf.article_id.isin(drop_source)]


# originally considered removing duplicates in entire corpus, but with date matching this may be too hard/messy/incorrect
#corpusdf_uniq = corpusdf[~corpusdf.article_id.isin(drop_all)]
#duplicates_dropped_entirecorpus = corpusdf[corpusdf.article_id.isin(drop_all)]

print("Number of articles in original cleaned corpus:", corpusdf.shape[0])
print("Number of articles in source-deduplicated corpus:", corpusdf_deduped_by_source.shape[0])
print("Number of articles in duplicates_dropped_by_source:", duplicates_dropped_by_source.shape[0])
#print("Number of articles in corpusdf_uniq", corpusdf_uniq.shape[0])
#print("Number of articles in duplicates_dropped_entirecorpus", duplicates_dropped_entirecorpus.shape[0])

# %% export them
pd.to_pickle(corpusdf_deduped_by_source, processeddatapath/'corpusdf_deduped_by_source.pickle')
# for topic modelling
corpusdf_deduped_by_source.to_csv(processeddatapath/'corpusdf_deduped_by_source.csv', index=False)

#pd.to_pickle(corpusdf_uniq, processeddatapath/'corpusdf_uniq.pickle')

# save the deduplication df
deduplication_df.to_csv(processeddatapath/'deduplication_df.csv', index=False)

# VISUALISE ALL OF THE DIFFERENCES ----------------------------------------
# %% for corpus by source
for_diffs_dropped_bysource_df = deduplication_df[
    (deduplication_df.source1==deduplication_df.source2) & 
    deduplication_df.docid1.isin(drop_source) &
    (deduplication_df.similarity >= dropping_similarity_cutoff)]

# mkdir if doesn't exist
pathlib.Path(cleandatapath/"dropped_same_source").mkdir(parents=True, exist_ok=True)

for_diffs_dropped_bysource_df.apply(lambda x: write_article_diffs(
    corpusdf=corpusdf, 
    article_kept_id = x.docid2, 
    article_dropped_id = x.docid1, 
    jaccard = '{:.3f}'.format(x.similarity),
    outdir=(cleandatapath/"dropped_same_source")), axis=1);

# %% entire corpus - what's would need to be dropped to fully de-dupe -----------------------------
for_diffs_dropped_corpus_df = deduplication_df[ 
    deduplication_df.docid1.isin(similar_across_sources) &
    (deduplication_df.similarity >= dropping_similarity_cutoff)]

# mkdir if doesn't exist
pathlib.Path(cleandatapath/"corpus_similar_across_sources").mkdir(parents=True, exist_ok=True)

for_diffs_dropped_corpus_df.apply(lambda x: write_article_diffs(
    corpusdf=corpusdf, 
    article_kept_id = x.docid2, 
    article_dropped_id = x.docid1, 
    jaccard ='{:.3f}'.format(x.similarity),
    outdir=(cleandatapath/"corpus_similar_across_sources")), axis=1);



# %% write out comparisons within source for similar but NOT dropped articles -----------------------------
# used for EDA and arriving at cutoffs
# get the similars:
# similar_source = set(get_duplicate_ids(
#     deduplication_df, 
#     sourcematch=True, 
#     similaritycutoff=similarity_cutoff)) - set(drop_source)

# for_diffs_undropped_similar = deduplication_df[
#     (deduplication_df.source1==deduplication_df.source2) & 
#     (deduplication_df.similarity < dropping_similarity_cutoff) &
#     (deduplication_df.similarity >=  similarity_cutoff) &
#     (deduplication_df.docid1.isin(similar_source))] 

# # mkdir if doesn't exist
# pathlib.Path(cleandatapath/"similar_same_source").mkdir(parents=True, exist_ok=True)

# for_diffs_undropped_similar.apply(lambda x: write_article_diffs(
#     corpusdf=corpusdf, 
#     article_kept_id = x.docid1, 
#     article_dropped_id = x.docid2, 
#     jaccard = round(x.similarity, 3),
#     outdir = (cleandatapath/"similar_same_source")), axis=1);

# %% how much did we drop?
def isidentical(value):
    if value == 1:
        return "identical"
    else:
        return "not identical"
for_diffs_dropped_bysource_df['identical'] = for_diffs_dropped_bysource_df.apply(lambda x: isidentical(x.similarity), axis=1)

for_diffs_dropped_corpus_df['identical'] = for_diffs_dropped_corpus_df.apply(lambda x: isidentical(x.similarity), axis=1)
# %%
# one was dropped
for_diffs_dropped_bysource_df.groupby(['identical', 'source1']).agg({'source1':'count'})
for_diffs_dropped_corpus_df.groupby(['identical', 'source1']).agg({'source1':'count'})


# %%
for_diffs_dropped_bysource_df.pivot_table(index='source1', columns='identical', fill_value=0, aggfunc='size').to_csv(cleandatapath/"dropped_same_source_counts.csv")
for_diffs_dropped_corpus_df.pivot_table(index='source1', columns='identical', fill_value=0, aggfunc='size').to_csv(cleandatapath/"similar_across_sources_counts.csv")

# %%

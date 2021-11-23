# %%
#import re
#import shutil
from datetime import datetime
import numpy as np
import pandas as pd
from pandas.core.algorithms import diff
from functs import  get_wordcount_from_metadata, get_record_from_corpus_df, obesitylist, count_words, get_record_by_article_id
import pathlib
from utils import get_projectpaths
(projectroot, rawdatapath, cleandatapath, processeddatapath) = get_projectpaths()
import seaborn as sns


# %%
corpusdf = pd.read_pickle(processeddatapath/'prepared_corpusdf.pickle')

# %% wordcount
# extract length from metadata
corpusdf = corpusdf.assign(wordcount_from_metatata=corpusdf['metadata'].map(lambda x: get_wordcount_from_metadata(x)))
# Get a better word count
# this has been superceded by spacy
corpusdf['wordcount_body'] = corpusdf['body'].apply(lambda x: count_words(x))
corpusdf['wordcount_title'] = corpusdf['title'].apply(lambda x: count_words(x))
corpusdf['wordcount_total'] = corpusdf.loc[:,['wordcount_body','wordcount_title']].sum(axis=1)


# %% count how many times words of interest appear in the body and title
for keyword in obesitylist():
    for place in ['title', 'body']:
        corpusdf[str(keyword + "_" + place + "_count")] = corpusdf[place].map(lambda x: x.lower().count(keyword))
# removing obesity's to prevent double-counting when doing obesity and obesity's
# obesogenic will also be counted when counting for obesogen
corpusdf['keywords_sum_body'] = corpusdf.loc[:, corpusdf.columns.str.endswith('_body_count')].sum(1) - corpusdf["obesity's_body_count"]
corpusdf['keywords_sum_title'] = corpusdf.loc[:, corpusdf.columns.str.endswith('_title_count')].sum(1) - corpusdf["obesity's_title_count"]
corpusdf['keywords_sum_total'] = corpusdf.keywords_sum_body + corpusdf.keywords_sum_title
corpusdf['keywords_sum_tara'] = corpusdf["obesity_body_count"] - corpusdf["obesity's_body_count"] + corpusdf["obese_body_count"] + corpusdf["obesity_title_count"] - corpusdf["obesity's_title_count"] + corpusdf["obese_title_count"]
# boolean as to match the UK corpus keyword count
corpusdf['obesity_boolean'] = corpusdf['keywords_sum_tara'] >= 3
corpusdf['obes_regex_count'] = corpusdf["body"].str.count('[Oo][Bb][Ee][Ss]+\w') + corpusdf["title"].str.count('[Oo][Bb][Ee][Ss]+\w')

differing_counts = corpusdf[corpusdf['obes_regex_count'] != corpusdf['keywords_sum_total']]
differing_counts.to_csv(projectroot/processeddatapath/'differing_counts.csv', index = False)

# %% plot

sns.scatterplot(data=corpusdf, x="wordcount_from_metatata", y="wordcount_total")



# %% most heavy outliers?
dataframeforviz = corpusdf.copy()
dataframeforviz['difference'] = dataframeforviz.apply(lambda x: 2 * abs(x['wordcount_from_metatata'] - x['wordcount_total'])/(x['wordcount_from_metatata'] + x['wordcount_total']), axis=1)
sns.histplot(data=dataframeforviz, x="difference");
dataframeforviz = dataframeforviz[dataframeforviz.difference > 0.2]


# %% get head
dataframeforviz.sort_values(by=['difference'], ascending=False)[['article_id','wordcount_from_metatata','wordcount_total', 'difference']]

# %% plot that
sns.histplot(data=dataframeforviz, x="difference");

# %%
import plotly.express as px
fig = px.scatter(dataframeforviz, x="wordcount_from_metatata", y="wordcount_total", hover_data=['article_id'])
fig.show()
# %% export different ones
(dataframeforviz[
    ['source', 'year', 'original_numeric_month','fourdigitcode','byline', 'date','article_id', 'title', 'wordcount_from_metatata', 'wordcount_total', 'difference', 'body']]).to_csv(processeddatapath/"wordcounts_different.csv", index=False)


# %% export cleaned corpus with word counts
# note no filtering has happened in this file, though we may want to later remove some crossword articles etc
pd.to_pickle(corpusdf, processeddatapath/'corpusdf_with_wc.pickle')
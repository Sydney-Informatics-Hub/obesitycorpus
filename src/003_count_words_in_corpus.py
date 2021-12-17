# %%
#import re
#import shutil
from datetime import datetime
import numpy as np
import pandas as pd
from pandas.core.algorithms import diff
from functs import  get_wordcount_from_metadata, get_record_from_corpus_df, obesitylist, count_words, get_record_by_article_id, sum_all_keywords
import pathlib
from utils import get_projectpaths
(projectroot, rawdatapath, cleandatapath, processeddatapath) = get_projectpaths()
import seaborn as sns
from nltk import tokenize
import re


# %%
corpusdf = pd.read_pickle(processeddatapath/'prepared_corpusdf.pickle')

# %% wordcount
# extract length from metadata
corpusdf = corpusdf.assign(wordcount_from_metatata=corpusdf['metadata'].map(lambda x: get_wordcount_from_metadata(x)))
# Get a better word count
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
corpusdf['obesity_boolean_ukcorpus'] = corpusdf['keywords_sum_tara'] >= 3
corpusdf['obes_regex_count'] = corpusdf["body"].str.count('[Oo][Bb][Ee][Ss]+\w') + corpusdf["title"].str.count('[Oo][Bb][Ee][Ss]+\w')

# %% first sentence
corpusdf['first_sent'] = corpusdf['body'].apply(lambda x: tokenize.sent_tokenize(
    # add a space before period in lowercaseletter.Capital to help NLTK
    re.sub(r'([a-z])\.([A-Z])', r'\1. \2', 
    # and digit.Capital , again to help NLTK
    re.sub(r'([0-9])\.([A-Z])', r'\1. \2', x)
    )
    )[0])

corpusdf['first_sent_count'] = corpusdf['first_sent'].apply(
    lambda x: sum_all_keywords(
        x, 
        # this makes sure that obesity's (and other plurals) is replaced with obesity and counted only once
        set(map(lambda x: x.replace("'s", ""), obesitylist()))
        ))

corpusdf['obesity_header_count'] = corpusdf['keywords_sum_title'] + corpusdf['first_sent_count']
corpusdf['obesity_boolean_header'] = corpusdf['obesity_header_count'] > 0

# %%
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
# note no filtering has happened in this file
pd.to_pickle(corpusdf, processeddatapath/'corpusdf_with_wc.pickle')
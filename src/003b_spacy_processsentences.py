# %%
import numpy as np
import pandas as pd
#import spacy
#from spacy.matcher import PhraseMatcher
#from spacy.tokens import Span
#from spacytextblob.spacytextblob import SpacyTextBlob
from functs import obesitylist, convert_month, explore_tokens
import pathlib
from utils import get_projectpaths
(projectroot, rawdatapath, cleandatapath, processeddatapath) = get_projectpaths()


# %% load data and obesity names
corpusdf = pd.read_pickle(str(cleandatapath/"corpusdf.pickle"))
# drop unneccessary columns
corpusdf = corpusdf.drop(['filename', 'encoding', 'confidence', 'fullpath','year', 'original_numeric_month'], axis=1)
# and make a key: date_source_fourdigitcode
sentencenlp = pd.read_pickle(processeddatapath/"sentencenlp.pkl")
obesitynames = obesitylist()

# %%
corpusdf = corpusdf.assign(sentencenlp=sentencenlp)
corpusdf = corpusdf.assign(articlesummary=corpusdf['sentencenlp'].map(lambda x: explore_tokens(x, obesitynames=obesitynames)))

# %%
lst_col = 'articlesummary'
# pivot article summary longer
corpusdf = pd.DataFrame({col:np.repeat(corpusdf[col].values, corpusdf[lst_col].str.len())for col in corpusdf.columns.difference([lst_col]) }).assign(**{lst_col:np.concatenate(corpusdf[lst_col].values)})[corpusdf.columns.tolist()]
# now each row has a dict with 7 keys
corpusdf = pd.concat([corpusdf.drop([lst_col], axis=1), corpusdf[lst_col].apply(pd.Series)], axis=1)
# and now that dict has been split into 7 columns

# %% clean up capitalisation
corpusdf["text"] = corpusdf["text"].str.lower()

# %% write out ---
corpusdf.drop(['body', 'metadata', 'sentencenlp'], axis=1).to_csv("../300_data_processed/pos_annotated_with_spacy.csv")

corpusdf.columns

# %% What is the overall structure of the dependency by words?
corpusdf.groupby(['text']).tag.value_counts()

corpusdf.groupby(['text']).dep.value_counts()

# %% Who is obese?
obese_mod = corpusdf[(corpusdf['amod']=="")]

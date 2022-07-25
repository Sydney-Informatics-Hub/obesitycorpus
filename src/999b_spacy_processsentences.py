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
corpusdf = pd.read_pickle(str(processeddatapath/'corpusdf_with_topics.pickle'))


# %%
bodiesnlp = pd.read_pickle(processeddatapath/"bodiesnlp.pkl")
titlesnlp = pd.read_pickle(processeddatapath/"titlesnlp.pkl")


obesitynames = obesitylist()

# %%
corpusdf = corpusdf.assign(bodiesnlp=bodiesnlp)


# %%
corpusdf = corpusdf.assign(bodiessummary=corpusdf['bodiesnlp'].map(lambda x: explore_tokens(x, obesitynames=obesitynames)))

# %%
lst_col = 'bodiessummary'
# pivot bodies summary longer
corpusdf = pd.DataFrame({col:np.repeat(corpusdf[col].values, corpusdf[lst_col].str.len())for col in corpusdf.columns.difference([lst_col]) }).assign(**{lst_col:np.concatenate(corpusdf[lst_col].values)})[corpusdf.columns.tolist()]
# now each row has a dict with 7 keys
corpusdf = pd.concat([corpusdf.drop([lst_col], axis=1), corpusdf[lst_col].apply(pd.Series)], axis=1)
# and now that dict has been split into 7 columns

# %% now create a copy df with only the variables of interest for the body
vars_to_keep = ['article_id', 'sentence', 'text', 'tag', 'dep', 'head', 'left','right']
corpusdf_body_postagged = corpusdf[vars_to_keep].copy()
# clean up capitalisation
corpusdf_body_postagged["text"] = corpusdf_body_postagged["text"].str.lower()


# %% now do the same for title
corpusdf = pd.read_pickle(str(processeddatapath/'corpusdf_with_topics.pickle'))

corpusdf = corpusdf.assign(titlesnlp=titlesnlp)
corpusdf = corpusdf.assign(titlessummary=corpusdf['titlesnlp'].map(lambda x: explore_tokens(x, obesitynames=obesitynames)))

#
lst_col = 'titlessummary'
# pivot bodies summary longer
corpusdf = pd.DataFrame({col:np.repeat(corpusdf[col].values, corpusdf[lst_col].str.len())for col in corpusdf.columns.difference([lst_col]) }).assign(**{lst_col:np.concatenate(corpusdf[lst_col].values)})[corpusdf.columns.tolist()]
# now each row has a dict with 7 keys
corpusdf = pd.concat([corpusdf.drop([lst_col], axis=1), corpusdf[lst_col].apply(pd.Series)], axis=1)
# and now that dict has been split into 7 columns


corpusdf_title_postagged = corpusdf[vars_to_keep].copy()
# clean up capitalisation
corpusdf_title_postagged["text"] = corpusdf_title_postagged["text"].str.lower()

# %% write out ---
corpusdf_body_postagged.to_csv(processeddatapath/"pos_bodies_annotated_with_spacy.csv")
corpusdf_title_postagged.to_csv(processeddatapath/"pos_titles_annotated_with_spacy.csv")



# %% What is the overall structure of the dependency by words?
corpusdf_body_postagged.groupby(['text']).tag.value_counts()

corpusdf_body_postagged.groupby(['text']).dep.value_counts()

# %% Who is obese?
#obese_mod = corpusdf[(corpusdf['amod']=="")]

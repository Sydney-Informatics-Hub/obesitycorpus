import numpy as np
import pandas as pd
#import spacy
#from spacy.matcher import PhraseMatcher
#from spacy.tokens import Span
#from spacytextblob.spacytextblob import SpacyTextBlob
from functs import obesitylist, convert_month, explore_tokens

# load data and obesity names
filesdf = pd.read_pickle("../200_data_clean/filesdf.pickle")
# drop unneccessary columns
filesdf = filesdf.drop(['filename', 'encoding', 'confidence', 'fullpath','year', 'numeric_month'], axis=1)
# and make a key: date_source_fourdigitcode
sentencenlp = pd.read_pickle("sentencenlp.pkl")
obesitynames = obesitylist()

filesdf = filesdf.assign(sentencenlp=sentencenlp)
filesdf = filesdf.assign(articlesummary=filesdf['sentencenlp'].map(lambda x: explore_tokens(x, obesitynames=obesitynames)))

lst_col = 'articlesummary'
# pivot article summary longer
filesdf = pd.DataFrame({col:np.repeat(filesdf[col].values, filesdf[lst_col].str.len())for col in filesdf.columns.difference([lst_col]) }).assign(**{lst_col:np.concatenate(filesdf[lst_col].values)})[filesdf.columns.tolist()]
# now each row has a dict with 7 keys
filesdf = pd.concat([filesdf.drop([lst_col], axis=1), filesdf[lst_col].apply(pd.Series)], axis=1)
# and now that dict has been split into 7 columns

# clean up capitalisation
filesdf["text"] = filesdf["text"].str.lower()

# write out ---
filesdf.drop(['body', 'metadata', 'sentencenlp'], axis=1).to_csv("../300_data_processed/pos_annotated_with_spacy.csv")

filesdf.columns

# What is the overall structure of the dependency by words?
filesdf.groupby(['text']).tag.value_counts()

filesdf.groupby(['text']).dep.value_counts()

# Who is obese?
obese_mod = filesdf[(filesdf['amod']=="")]

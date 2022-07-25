# %%
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacytextblob.spacytextblob import SpacyTextBlob
import pickle
import functs as f
import pathlib
from utils import get_projectpaths
(projectroot, rawdatapath, cleandatapath, processeddatapath) = get_projectpaths()

# %%
corpusdf = pd.read_pickle(str(processeddatapath/'corpusdf_with_topics.pickle'))

# %%
nlp = spacy.load("en_core_web_sm")
obesitynames = ['obesity', 'obese', 'obesogen']
patterns = [nlp(text) for text in obesitynames]
phrase_matcher = PhraseMatcher(nlp.vocab)
phrase_matcher.add('obes', None, *patterns)
nlp.add_pipe('spacytextblob')

# %%
bodyinput = [nlp(x.strip().lower())for x in corpusdf.body.to_list()]

# %% same for titles
titleinput = [nlp(x.strip().lower())for x in corpusdf.title.to_list()]


# %%

def whatIwantSpacyDoc(spacyinput):
    sentencenlp = []
    for doc in spacyinput:
        sentencelist = []
        for sent in doc.sents:
            for match_id, start, end in phrase_matcher(nlp(sent.text)):
                sentencelist.append(sent.text)
        mynlp = [nlp(x.strip()) for x in sentencelist]
        sentencenlp.append(mynlp)
    return sentencenlp

# %% for bodies
bodiesnlp = whatIwantSpacyDoc(bodyinput)

# %% for titles
titlesnlp = whatIwantSpacyDoc(titleinput)

# %%
with open(str(processeddatapath/'bodiesnlp.pkl'), 'wb') as file:
    pickle.dump(bodiesnlp, file)

# %%
with open(str(processeddatapath/'titlesnlp.pkl'), 'wb') as file:
    pickle.dump(titlesnlp, file)
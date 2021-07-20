import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacytextblob.spacytextblob import SpacyTextBlob
import pickle
import functs as f


filesdf = pd.read_pickle("../200_data_clean/filesdf.pickle")

nlp = spacy.load("en_core_web_sm")
obesitynames = f.obesitylist()
patterns = [nlp(text) for text in obesitynames]
phrase_matcher = PhraseMatcher(nlp.vocab)
phrase_matcher.add('obes', None, *patterns)
nlp.add_pipe('spacytextblob')

spacyinput = [nlp(x.strip()) for x in filesdf.body.to_list()]

sentencenlp = []
for doc in spacyinput:
    sentencelist = []
    for sent in doc.sents:
        for match_id, start, end in phrase_matcher(nlp(sent.text)):
            sentencelist.append(sent.text)
    mynlp = [nlp(x.strip()) for x in sentencelist]
    sentencenlp.append(mynlp)

with open('sentencenlp.pkl', 'wb') as file:
    pickle.dump(sentencenlp, file)
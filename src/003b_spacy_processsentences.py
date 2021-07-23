import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacytextblob.spacytextblob import SpacyTextBlob
import pickle
import functs as f

# load data and obesity names
filesdf = pd.read_pickle("../200_data_clean/filesdf.pickle")
sentencenlp = pd.read_pickle("sentencenlp.pkl")
obesitynames = f.obesitylist()


filesdf = filesdf.assign(sentencenlp = sentencenlp)
filesdf = filesdf.assign(articlesummary = filesdf['sentencenlp'].map(lambda x: f.explore_tokens(x, obesitynames = obesitynames)))


filesdf.articlesummary[0]

tmp = filesdf.head().copy()


def single_word_tokensummary(articlesummary, word):
    taglist = []
    dep = []
    heads = []
    left = []
    right = []
    for x in articlesummary:
        if x[0]['text'] == word:
            taglist.append(x[0]['tag'])
            dep.append(x[0]['dep'])
            heads.append(x[0]['head'])
            left.append(x[0]['left'])
            right.append(x[0]['right'])
    return {
        'tag': taglist, 
        'dep': dep,
        'heads': heads,
        'left': left,
        'right': right}

tmp2 = tmp.copy()[['outputfilename', 'year','short_month','title', 'byline', 'articlesummary']]
tmp2['file'] = tmp2['outputfilename'].str.replace('../200_data_clean/', '', regex=False).str.replace('.txt', '', regex=False)



for word in obesitynames:
    tmp2[word] = tmp2['articlesummary'].map(lambda x: single_word_tokensummary(articlesummary = x, word = word))




# for doc in spacytest:
#     for chunk in doc.noun_chunks:
#         if chunk.root.text == "obesity":
#             print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)



#print(doc._.polarity)
#print(doc._.subjectivity)





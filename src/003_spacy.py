import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacytextblob.spacytextblob import SpacyTextBlob
from spacy import displacy
import pickle

filesdf = pd.read_pickle("../200_data_clean/filesdf.pickle")

nlp = spacy.load("en_core_web_sm")
obesitynames = ['obesity', 'obese', "obesogenic", "obesogen"]
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


filesdf = filesdf.assign(sentencenlp = sentencenlp)



filesdf = filesdf.assign(articlesummary = filesdf['sentencenlp'].map(lambda x: f.explore_tokens(x)))


filesdf.articlesummary[0]

tmp = filesdf.head().copy()

tmp2 = tmp.articlesummary[0]

mydict = {}
# for word in obesitynames:
#     mydict[word]: {'tag': }




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
    return({
        'tag': taglist, 
        'dep': dep,
        'heads': heads,
        'left': left,
        'right': right}
    )


for word in obesitynames:
    tmp['articlesummary'].map(lambda x: single_word_tokensummary(articlesummary = x, word = word))




# for doc in spacytest:
#     for chunk in doc.noun_chunks:
#         if chunk.root.text == "obesity":
#             print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)



#print(doc._.polarity)
#print(doc._.subjectivity)



tmp = pd.read_pickle("sentencenlp.pkl")

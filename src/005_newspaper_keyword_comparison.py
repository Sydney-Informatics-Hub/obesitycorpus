import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pathlib
from utils import get_project_root

projectroot = str(get_project_root())
rawdatapath = pathlib.Path(projectroot + "/100_data_raw/Australian Obesity Corpus")
cleandatapath = pathlib.Path(projectroot + "/200_data_clean")
processeddatapath = pathlib.Path(projectroot + "/300_data_processed")

corpusdf = pd.read_pickle(str(cleandatapath) + "/corpusdf.pickle")

corpusdict = {}
for source in corpusdf.source.unique().tolist():
    body = corpusdf[corpusdf['source'] == source]['body'].to_list()
    title = corpusdf[corpusdf['source'] == source]['title'].to_list()
    bodystring = ' '.join([str(elem) for elem in body])
    titlestring = ' '.join([str(elem) for elem in title])
    corpusdict[source] = bodystring + " " + titlestring

corpusdf_counting = pd.DataFrame.from_dict(corpusdict, orient='index', columns=['text'])
corpusdf_counting['source'] = corpusdf_counting.index

vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
corpustokencounts = vectorizer.fit_transform(corpusdf_counting['text'].values)

wordcountdf = pd.DataFrame(data=np.transpose(corpustokencounts.toarray()), columns=corpusdf.source.unique().tolist())
wordcountdf['word'] = vectorizer.get_feature_names()

total_by_source = wordcountdf.loc[:, wordcountdf.columns != "word"].sum(axis=0).to_dict()
total_word_used = wordcountdf.loc[:, wordcountdf.columns != "word"].sum(axis=1)
wordcountdf['total_word_used'] = total_word_used
# wordcountdf.to_csv("wordcounts.csv")
total_words_in_corpus = sum(total_word_used)

for source in corpusdf.source.unique().tolist():
    wordcountdf[str('expected_wc_'+source)] = total_by_source[source] * wordcountdf['total_word_used'] / total_words_in_corpus

def single_source_ln(source_wc, expected_wc_source):
    if (source_wc == 0) or (expected_wc_source == 0):
        return 0
    else:
        # unlike the excel spreadsheet put the 2x multiplication here
        return 2 * np.log(source_wc/expected_wc_source)

for source in corpusdf.source.unique().tolist():
    if source == corpusdf.source.unique().tolist()[0]:
        # first source
        tmparray =  wordcountdf.apply(lambda x: single_source_ln(x[source], x[str('expected_wc_' + source)]), axis=1).array
    else:
        tmparray += wordcountdf.apply(lambda x: single_source_ln(x[source], x[str('expected_wc_' + source)]), axis=1).array

wordcountdf['Log Likelihood'] = tmparray


degrees_of_freedom = len(corpusdf.source.unique().tolist()) - 1
wordcountdf['Bayes Factor BIC'] = wordcountdf['Log Likelihood'] - (degrees_of_freedom * np.log(total_words_in_corpus))

wordcountdf['ELL'] = wordcountdf['Log Likelihood']/(total_words_in_corpus * np.log(wordcountdf.filter(regex='expected_wc').min(axis=1)))
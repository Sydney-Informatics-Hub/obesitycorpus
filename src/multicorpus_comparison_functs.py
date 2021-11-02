import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from pandas._testing import assert_frame_equal

def collapse_corpus_by_source(df):
    '''
    Takes a corpus df with title, body & source columns, one article per row
    Returns a df with n_source rows, where 'text' contains the union of body and title for
    all articles from that source
    '''
    corpusdict = {}
    for source in df.source.unique().tolist():
        body = df[df['source'] == source]['body'].to_list()
        title = df[df['source'] == source]['title'].to_list()
        bodystring = ' '.join([str(elem) for elem in body])
        titlestring = ' '.join([str(elem) for elem in title])
        corpusdict[source] = bodystring + " " + titlestring
    df_counting = pd.DataFrame.from_dict(corpusdict, orient='index', columns=['text'])
    df_counting['source'] = df_counting.index
    return df_counting

def count_words(df):
    '''
    Takes a df with one row per source, with cols 'source' and 'text'
    Returns a df of word counts from each source in the corpus
    '''
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    corpustokencounts = vectorizer.fit_transform(df['text'].values)
    wordcountdf = pd.DataFrame(data=np.transpose(corpustokencounts.toarray()), columns=df.source.unique().tolist())
    wordcountdf['word'] = vectorizer.get_feature_names()
    return wordcountdf

def get_totals(df):
    '''
    Takes a word count df with counts for each source plus 'word'
    Returns df with sum by source plus dict total_by_source & int total_words_in_corpus
    '''
    total_by_source = df.loc[:, df.columns != "word"].sum(axis=0).to_dict()
    total_word_used = df.loc[:, df.columns != "word"].sum(axis=1)
    df['total_word_used'] = total_word_used
    total_words_in_corpus = sum(total_word_used)
    return (df, total_by_source, total_words_in_corpus)

def single_source_ln(source_wc, expected_wc_source):
    if (source_wc == 0) or (expected_wc_source == 0):
        return 0
    else:
        # unlike the excel spreadsheet put the 2x multiplication here
        return 2 * source_wc * np.log(source_wc/expected_wc_source)

def get_percent_diff(normalised_wc_source, normalised_restofcorpus_wc,diff_zero_freq_adjustment):
    if normalised_restofcorpus_wc == 0:
        divideby = diff_zero_freq_adjustment
    else:
        divideby = normalised_restofcorpus_wc
    return 100 * (normalised_wc_source - normalised_restofcorpus_wc)/divideby

def log2_ratio(normalised_freq_source, normalised_freq_rest_of_corpus, total_words_source1, total_words_rest_of_corpus):
    # constant as per spreadsheet
    log_ratio_zero_freq_adjustment = 0.5
    numerator = normalised_freq_source if normalised_freq_source != 0 else log_ratio_zero_freq_adjustment/total_words_source1
    denominator = normalised_freq_rest_of_corpus if normalised_freq_rest_of_corpus != 0 else log_ratio_zero_freq_adjustment/total_words_rest_of_corpus
    return np.log2(numerator/denominator)

def odds_ratio(source_wc, rest_of_corpus_wc, total_words_source, total_words_rest_of_corpus):
    numerator = source_wc/(total_words_source-source_wc)
    denominator = rest_of_corpus_wc/(total_words_rest_of_corpus-rest_of_corpus_wc)
    if denominator == 0:
        return np.nan
    else:
        return numerator/denominator

def relative_risk(normalised_wc, normalised_restofcorpus_wc):
    if normalised_restofcorpus_wc == 0:
        return np.nan
    else:
        return normalised_wc/normalised_restofcorpus_wc


def two_corpus_compare(df, total_by_source, total_words_in_corpus):
    '''
    Compare two corpora, as per the 2 corpus Lancaster example
    Works on two corpora
    '''
    outdf = df.copy()
    sources = outdf.columns.difference(['word', 'total_word_used'])
    # comparing two, so df=1
    degrees_of_freedom = 1
    diff_zero_freq_adjustment = 1E-18

    for source in sources:
        wc_restofcorpus = total_words_in_corpus - total_by_source[source]
        outdf[str('expected_wc_'+source)] = total_by_source[source] * outdf['total_word_used']/ total_words_in_corpus
        outdf[str('expected_restofcorpus_wc_'+source)] = wc_restofcorpus * outdf['total_word_used']/total_words_in_corpus
        outdf[str('normalised_wc_'+source)] = outdf[source]/total_by_source[source]
        outdf[str('normalised_restofcorpus_wc_'+source)] = (outdf['total_word_used'] - outdf[source])/wc_restofcorpus
        outdf[str('overuse_'+source)] = outdf[str('normalised_wc_'+source)] > outdf[str('normalised_restofcorpus_wc_'+source)]
        # log-likelihood calculation per-corpus vs rest of corpus
        tmparray = outdf.apply(lambda x: single_source_ln(x[source], x[str('expected_wc_' + source)]), axis=1).array
        tmparray += outdf.apply(lambda x: single_source_ln((x['total_word_used'] - x[source]), x[str('expected_restofcorpus_wc_' + source)]), axis=1).array
        outdf[str('log_likelihood_'+source)] = tmparray
        outdf[str('percent_diff_'+source)] = outdf.apply(lambda x: get_percent_diff(x[str('normalised_wc_'+source)],x[str('normalised_restofcorpus_wc_'+source)], diff_zero_freq_adjustment), axis=1)
        outdf['bayes_factor_bic_'+source] = outdf[str('log_likelihood_'+source)] - (degrees_of_freedom * np.log(total_words_in_corpus))
        outdf['ell_'+source] = outdf[str('log_likelihood_'+source)]/(total_words_in_corpus * np.log(outdf.filter(regex=str('expected.*' + source )).min(axis=1)))
        outdf['relative_risk_'+source] = outdf.apply(lambda x: relative_risk(x[str('normalised_wc_'+source)], x[str('normalised_restofcorpus_wc_'+source)]), axis=1)
        outdf['log_ratio_' + source] = outdf.apply(lambda x: log2_ratio(x[str('normalised_wc_'+source)], x[str('normalised_restofcorpus_wc_'+source)], total_by_source[source], wc_restofcorpus), axis=1)
        outdf['odds_ratio_' + source] = outdf.apply(lambda x: odds_ratio(x[source], (x['total_word_used'] - x[source]), total_by_source[source], wc_restofcorpus), axis=1)
    return outdf


def test_twocorpus_compare(projectroot):
    # test file has raw data and results
    test2 = pd.read_csv(str(projectroot) + "/100_data_raw/211008_SigEff_2test.csv")
    # keep only word counts and the word in there
    test2_wc_only = test2[test2.columns[pd.Series(test2.columns).str.startswith('corpus')]].copy()
    test2_wc_only['word'] = test2['word']
    testcountdf, total_by_source_test, total_words_in_corpus_test = get_totals(df=test2_wc_only)
    twocorp_comparison_test = two_corpus_compare(df=testcountdf, total_by_source=total_by_source_test, total_words_in_corpus=total_words_in_corpus_test)
    # the assertion passes :)
    return assert_frame_equal(twocorp_comparison_test[test2.columns.to_list()], test2, check_exact=False, atol=0.1, rtol=0.1)





def n_corpus_compare(df, total_by_source, total_words_in_corpus):
    '''
    Compare multiple corpora, as per the 6 corpus Lancaster example
    Works on 2+ - infinity corpora correctly
    '''
    outdf = df.copy()
    sources = outdf.columns.difference(['word', 'total_word_used'])
    for source in sources:
        outdf[str('expected_wc_'+source)] = total_by_source[source] * outdf['total_word_used'] / total_words_in_corpus

    for source in sources:
        if source == sources[0]:
            # first source
            tmparray = outdf.apply(lambda x: single_source_ln(x[source], x[str('expected_wc_' + source)]), axis=1).array
        else:
            tmparray += outdf.apply(lambda x: single_source_ln(x[source], x[str('expected_wc_' + source)]), axis=1).array

    outdf['Log Likelihood'] = tmparray
    degrees_of_freedom = len(sources) - 1
    outdf['Bayes Factor BIC'] = outdf['Log Likelihood'] - (degrees_of_freedom * np.log(total_words_in_corpus))
    outdf['ELL'] = outdf['Log Likelihood']/(total_words_in_corpus * np.log(outdf.filter(regex='expected_wc').min(axis=1)))
    return outdf

def test_multicorpus_compare(projectroot):
    # test file has raw data and results
    test6 = pd.read_csv(str(projectroot) + "/100_data_raw/211008_SigEff_6test.csv")
    # keep only word counts and the word in there
    test6_wc_only = test6[test6.columns[pd.Series(test6.columns).str.startswith('corpus')]].copy()
    test6_wc_only['word'] = test6['word']
    testcountdf, total_by_source_test, total_words_in_corpus_test = get_totals(df=test6_wc_only)
    multicorp_comparison_test = n_corpus_compare(df=testcountdf, total_by_source=total_by_source_test, total_words_in_corpus=total_words_in_corpus_test)
    # the assertion passes :)
    return assert_frame_equal(multicorp_comparison_test, test6)
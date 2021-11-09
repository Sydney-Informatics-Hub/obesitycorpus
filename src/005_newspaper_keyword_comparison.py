import pandas as pd
import pathlib
from utils import get_project_root
from multicorpus_comparison_functs import collapse_corpus_by_source, count_words, get_totals
from multicorpus_comparison_functs import two_corpus_compare, test_twocorpus_compare
from multicorpus_comparison_functs import n_corpus_compare, test_multicorpus_compare

# The documentation for some of this is also here;
# http://ucrel.lancs.ac.uk/llwizard.html

projectroot = str(get_project_root())
rawdatapath = pathlib.Path(projectroot + "/100_data_raw/Australian Obesity Corpus")
cleandatapath = pathlib.Path(projectroot + "/200_data_clean")
processeddatapath = pathlib.Path(projectroot + "/300_data_processed")

# corpusdf must have title, body and source columns
corpusdf = pd.read_pickle(str(cleandatapath) + "/corpusdf.pickle")
corpusdf_forcounting = collapse_corpus_by_source(df=corpusdf)
wordcountdf = count_words(df=corpusdf_forcounting)

# get rowsums and colsums for downstream calculations
wordcountdf, total_by_source, total_words_in_corpus = get_totals(df=wordcountdf)

# one corpus against the rest, pairwise for each  ---
# test first (on lancaster sample snippets)
test_twocorpus_compare(projectroot=projectroot)
# then apply on our corpus
pairwise_compare = two_corpus_compare(wordcountdf, total_by_source, total_words_in_corpus)

# multiple corpus comparisons ---
# test first (on lancaster sample snippets)
test_multicorpus_compare(projectroot=projectroot)
# then apply on our corpus
multicorp_comparison = n_corpus_compare(df=wordcountdf, total_by_source=total_by_source, total_words_in_corpus=total_words_in_corpus)

# Export to csv
pairwise_compare.to_csv(str(processeddatapath/"pairwise_comparisons.csv"), index=False)
multicorp_comparison.to_csv(str(processeddatapath/"multicorp_comparison.csv"), index=False)

# Compare by publisher
mastheads = pd.read_csv(str(projectroot) + "/100_data_raw/MastheadOwners.csv")

mastheads_dict = {}
for corporation in mastheads.corporation.unique().tolist():
    mastheads_dict[corporation] = (wordcountdf[
        # grab the publications from each corporation from the mastheads df as a list
        # use that to grab the correct columns from the wordcount df (1st line)
                                       mastheads[mastheads['corporation'] == corporation].source.to_list()]
    # and find the rowsum aka count per masthead
                                   .sum(axis = 1))

# one row per word!
mastheads_dict['word'] = wordcountdf['word']

mastheads_wordcount = pd.DataFrame.from_dict(mastheads_dict)

# get rowsums and colsums for downstream calculations
mastheads_wordcount, total_by_masthead, total_words_in_corpus2 = get_totals(df=mastheads_wordcount)
# compare mastheads vs rest
pairwise_compare_masthead = two_corpus_compare(mastheads_wordcount, total_by_masthead, total_words_in_corpus2)
# export to csv
pairwise_compare_masthead.to_csv(str(processeddatapath/"pairwise_compare_masthead.csv"), index=False)

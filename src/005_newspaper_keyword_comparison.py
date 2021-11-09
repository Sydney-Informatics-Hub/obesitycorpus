import pandas as pd
import pathlib
from utils import get_project_root
from multicorpus_comparison_functs import collapse_corpus_by_source, count_words, get_totals
from multicorpus_comparison_functs import two_corpus_compare, test_twocorpus_compare
from multicorpus_comparison_functs import n_corpus_compare, test_multicorpus_compare

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
pairwise_compare.to_csv(str(processeddatapath/"pairwise_comparisons.csv"))
multicorp_comparison.to_csv(str(processeddatapath/"multicorp_comparison.csv"))
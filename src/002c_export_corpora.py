# %%
import re
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
from functs import clean_page_splits, clean_redundant_phrases, strip_newlines, apply_to_titlebody, abbreviate_source , obesitylist
from functs import clean_wa
from functs import get_byline, parse_filename, readfilesin
from functs import get_text4digitcode, clean_nonascii ,clean_quotes , get_date ,clean_quot, replace_six_questionmarks, get_record_from_corpus_df
from functs import write_corpus_titlebody, write_corpus_sketchengine, get_wordcount_from_metadata, write_corpus_cqpweb, write_corpus_nested
import pathlib
from utils import get_projectpaths
(projectroot, rawdatapath, cleandatapath, processeddatapath) = get_projectpaths()

# imports for debugging non-unicode characters
# from functs import find_problems, find_filename_from_string, find_specific_character_with_preceding, 
# from functs import print_body_from_string, find_specific_character_wout_preceding, 
# from functs import where_is_byline

# %%
corpusdf = # pd.read_csv(str(processeddatapath / 'inferred_dataset_encodings.csv'))




# just including articles with three or more mentions in article (either body or headline)
# just articles that include a mention in the nucleus {headline or first sentence}

# todo write some wordcounts for each corpus
# write_corpus_summary_tables(corpusdf, path=cleandatapath, articlecounts_name="articlecounts", wordcounts_name="wordcounts")
# same for other two corpus versions

# %% write corpus out as per client's request
write_corpus_titlebody(df=corpusdf, cleandatapath=cleandatapath, directoryname="corpus-titlebody")
write_corpus_cqpweb(df=corpusdf, cleandatapath=cleandatapath, directoryname="corpus-cqpweb")
write_corpus_sketchengine(df=corpusdf, cleandatapath=cleandatapath, directoryname="corpus-sketchengine")
write_corpus_nested(df=corpusdf, cleandatapath=cleandatapath, directoryname="corpus-nested")

# %% export in data science formats
# TODO before export make sure Unnamed: 0	 this column doesn't exist
corpusdf.to_pickle(cleandatapath/'corpusdf.pickle')
corpusdf.to_csv(cleandatapath/"corpusdf.csv", index=False)

# copy to cloudstor
# very fragile local link :(
shutil.copytree(cleandatapath, '../../../../cloudstor/projects/obesity', dirs_exist_ok=True)


# %%

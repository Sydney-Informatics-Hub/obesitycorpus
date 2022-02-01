# %%
import re
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
from functs import write_corpus_summary_tables, get_record_by_article_id

from functs import write_corpus_titlebody, write_corpus_sketchengine,  write_corpus_cqpweb, write_corpus_nested
import pathlib
from utils import get_projectpaths
(projectroot, rawdatapath, cleandatapath, processeddatapath) = get_projectpaths()

# %%
corpusdf = pd.read_pickle(processeddatapath/'corpusdf_with_topics.pickle')

# %%
# generate a corpus with 3+ counts of obesity, as per tara
corpusdf_tara = corpusdf.copy()
corpusdf_tara = corpusdf_tara[corpusdf_tara['obesity_boolean_ukcorpus'] == True].reset_index(drop = True)


# %%
# generate a corpus with 1+ count of obesity in the title or first sentence
corpusdf_header = corpusdf.copy()
corpusdf_header = corpusdf_header[corpusdf_header['obesity_boolean_header'] == True].reset_index(drop = True)

# %% write full corpus out as per client's request
write_corpus_titlebody(df=corpusdf, cleandatapath=cleandatapath, directoryname="corpus_titlebody")
write_corpus_cqpweb(inputdf=corpusdf, cleandatapath=cleandatapath, directoryname="corpus_cqpweb", write_actual_files=True)
write_corpus_sketchengine(inputdf=corpusdf, cleandatapath=cleandatapath, directoryname="corpus_sketchengine")
write_corpus_nested(df=corpusdf, cleandatapath=cleandatapath, directoryname="corpus_nested")

# %% 3+ mentions
write_corpus_titlebody(df=corpusdf_tara, cleandatapath=cleandatapath, directoryname="corpus_titlebody_3plus")
write_corpus_sketchengine(inputdf=corpusdf_tara, cleandatapath=cleandatapath, directoryname="corpus_sketchengine_3plus")
write_corpus_nested(df=corpusdf_tara, cleandatapath=cleandatapath, directoryname="corpus_nested_3plus")
write_corpus_cqpweb(inputdf=corpusdf, cleandatapath=cleandatapath, directoryname="corpus_3plus", write_actual_files=False)


# %% in header (1 sentence & title)
write_corpus_titlebody(df=corpusdf_header, cleandatapath=cleandatapath, directoryname="corpus_titlebody_header")
write_corpus_sketchengine(inputdf=corpusdf_header, cleandatapath=cleandatapath, directoryname="corpus_sketchengine_header")
write_corpus_nested(df=corpusdf_header, cleandatapath=cleandatapath, directoryname="corpus_nested_header")
write_corpus_cqpweb(inputdf=corpusdf, cleandatapath=cleandatapath, directoryname="corpus_header", write_actual_files=False)

# %% write for testing
#corpusdf_mini = corpusdf.groupby('source').head(15).reset_index(drop=True)
#write_corpus_cqpweb(inputdf=corpusdf_mini, cleandatapath=cleandatapath, directoryname="corpus_cqpweb-mini")




# just including articles with three or more mentions in article (either body or headline)
write_corpus_summary_tables(
    corpusdf=corpusdf, 
    cleandatapath=cleandatapath, 
    articlecounts_name="articlecounts_full", 
    wordcounts_name="wordcounts_full")
# same for tara corpus version
write_corpus_summary_tables(
    corpusdf=corpusdf_tara, 
    cleandatapath=cleandatapath, 
    articlecounts_name="articlecounts_3plus", 
    wordcounts_name="wordcounts_3plus")
# and for header corpus version
write_corpus_summary_tables(
    corpusdf=corpusdf_header, 
    cleandatapath=cleandatapath, 
    articlecounts_name="articlecounts_header", 
    wordcounts_name="wordcounts_header")

# %% export in data science formats
#corpusdf.to_pickle(cleandatapath/'corpusdf_final.pickle')
#corpusdf.to_csv(cleandatapath/"corpusdf_final.csv", index=False)

# %% copy to cloudstor
# very fragile local link :(
# shutil.copytree(cleandatapath, '../../../../cloudstor/projects/obesity', dirs_exist_ok=True)



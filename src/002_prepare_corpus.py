# %%
import re
#import shutil
from datetime import datetime
import numpy as np
import pandas as pd
from functs import clean_page_splits, clean_redundant_phrases, strip_newlines, apply_to_titlebody, abbreviate_source 
from functs import clean_wa, replace_triple_quote
from functs import get_byline, parse_filename, readfilesin
from functs import get_text4digitcode, clean_nonascii ,clean_quotes , get_date ,clean_quot, replace_six_questionmarks
from functs import get_record_from_corpus_df, remove_australian_authordeets
import pathlib
from utils import get_projectpaths
(projectroot, rawdatapath, cleandatapath, processeddatapath) = get_projectpaths()

# imports for debugging non-unicode characters
# from functs import find_problems, find_filename_from_string, find_specific_character_with_preceding, 
# from functs import print_body_from_string, find_specific_character_wout_preceding, 
# from functs import where_is_byline

# %%
corpusdf = pd.read_csv(str(processeddatapath / 'inferred_dataset_encodings.csv'))

# get column with files as full path
corpusdf['fullpath'] = str(str(rawdatapath) + "/") + corpusdf['filename'].astype(str)
print(corpusdf.shape[0])

result = [readfilesin(x, y) for x, y in zip(corpusdf['fullpath'], corpusdf['encoding'])]

corpusdf['contents'] = result

# %% filter out those files that don't have the word "body" (case-insensitive) in them
# These are the list files and the zip file
# write out the metadata for them for records
dropped = corpusdf[~corpusdf['contents'].str.contains('body', case=False)]
dropped.to_csv(projectroot/cleandatapath/'dropped.csv', index=False)
# then filter them out
corpusdf = corpusdf[corpusdf['contents'].str.contains('body', case=False)]

# %% split header and body, thankfully case-sensitive works
corpusdf['contents_split'] = corpusdf['contents'].str.split("\nBody\n")
# always splits into two
# corpusdf['contents_len'] = corpusdf['contents_split'].str.len()
# take advantage of this:
corpusdf[['metadata', 'body']] = pd.DataFrame(corpusdf.contents_split.tolist(), index=corpusdf.index)
corpusdf = corpusdf.drop(columns=['contents_split', 'contents', 'fullpath'])

# grab the best pass at the title, and convert all so only the first word is capitalised
corpusdf = corpusdf.assign(title=corpusdf['metadata'].map(lambda x: x.partition('\n')[0].capitalize()))

# %% get date/month from the relevant directory of the filepath
filename_fields = corpusdf["filename"].apply(parse_filename)
corpusdf = corpusdf.assign(**filename_fields)
# extract the four digit code from the filename
corpusdf = corpusdf.assign(fourdigitcode=corpusdf['filename'].map(get_text4digitcode))

# %% where are the bylines?
# code used for eda but not needed later
# corpusdf = corpusdf.assign(byline_locs = corpusdf['contents_split'].map(lambda x: where_is_byline(x)))
# corpusdf['bylines_flat'] = [','.join(map(str, l)) for l in corpusdf['byline_locs']]
corpusdf = corpusdf.assign(byline=corpusdf['metadata'].map(lambda x: get_byline(x)))
# lots of room to clean bylines if needed/desired...

# %% the below results in an empty set
assert not set([item for sublist in [re.findall(r'[^\x00-\x7F]+',x) for x in corpusdf[corpusdf.encoding == "ascii"].body.tolist()]  for item in sublist])

# %% Clean up ---
# clean up non-unicode characters
apply_to_titlebody(corpusdf, clean_nonascii)
corpusdf["metadata"] = corpusdf["metadata"].apply(clean_nonascii)

# get rid of newlines at start and end of body & title
apply_to_titlebody(corpusdf, strip_newlines)
# clean up quotes
apply_to_titlebody(corpusdf, clean_quotes)
# and the random repeats of "&quot;"
apply_to_titlebody(corpusdf, clean_quot)
# and replace six ?????? specifically with double quotes...
corpusdf["body"] = corpusdf["body"].apply(replace_six_questionmarks)
# and replace triple quotes in titles
corpusdf['title'] = corpusdf['title'].apply(replace_triple_quote)
# and replace multiple dashes with just one
corpusdf["body"] = corpusdf["body"].apply(lambda x: (re.sub(r'-+', '-', x)))
# replace some odd Western Australia quote/text box, observed in the titles
apply_to_titlebody(corpusdf, clean_wa)
# clean page references and social media references
corpusdf["body"] = corpusdf["body"].apply(clean_page_splits)
corpusdf["body"] = corpusdf["body"].apply(clean_redundant_phrases)
# clean publication-specific issues

#%%
# this was not used, as could remove meaningful text as well
# corpusdf["body"] = corpusdf["body"].apply(remove_australian_authordeets)

# %% get dates
corpusdf['date'] = corpusdf["metadata"].apply(get_date)
# some may not parse
print("The total number of files where we were unable to find a date in the metadata was ", sum(corpusdf.date.isna()), "files.")
print("These will be replaced from the filename, setting the date to the 1st of the month.")
# if unable to be detected replace with the first of the month and year that exists in the filename
# at least for this corpus - 1 file, no date in raw file
tmpdate = corpusdf.apply(lambda x: (datetime.strptime(str(x['year'] + "-" + x['original_numeric_month'] + "-01 00:00:00"), "%Y-%m-%d %H:%M:%S")), axis=1)
corpusdf['date'] = corpusdf['date'].fillna(tmpdate)
# get the month from metadata and use it (as it's correct for articles based on inspection)
corpusdf['month_metadata'] = corpusdf.date.dt.month
corpusdf['month_metadata'] = corpusdf['month_metadata'].apply(lambda x: str(x).zfill(2))

# %% test that dates from filenames and dates from contents match
years_match = pd.to_numeric(corpusdf.year).equals(corpusdf.date.dt.year)
months_match = pd.to_numeric(corpusdf.original_numeric_month).equals(corpusdf.date.dt.month)
assert years_match
unmatching_months = corpusdf[pd.to_numeric(corpusdf.original_numeric_month) != corpusdf.date.dt.month]
unmatching_months.to_csv(projectroot/cleandatapath/"unmatching_months.csv", index=False)

# can use the below functions to narrow down issues ---
# find_problems(start, end, colname = "body", corpusdf = corpusdf)
# find_specific_character_with_preceding(character, start, end, colname = "body", corpusdf = corpusdf)
# find_specific_character_wout_preceding(character, start, end, colname = "body", corpusdf = corpusdf)
# find_filename_from_string(string, corpusdf = corpusdf)
# print_body_from_string(string, corpusdf = corpusdf)

# %% generate an article_id as per cqpweb specifications/Andrew Hardie email
corpusdf['shortcode'] = abbreviate_source(corpusdf["source"])
# make a padded rownumber column so max is the same as number of articles in the FINAL corpus
corpusdf['rownumber'] = np.arange(len(corpusdf)).astype(str)
corpusdf['rownumber'] = corpusdf['rownumber'].str.zfill(5)
# and a full id for each article - ex GU0801004
# note using the "real" month from the metadata
# even if there are dupes the rownumber will keep these still unique
corpusdf['article_id'] =  corpusdf['shortcode'] + corpusdf['year'].apply(lambda x: x[2:]) + corpusdf['month_metadata'] + corpusdf['rownumber']

# remove redundant columns
corpusdf = corpusdf.drop(columns=['filename','encoding', 'confidence'])

# remove crossword puzzles
print("The total number of crossword puzzles was ", len(corpusdf[corpusdf.body.str.contains('^ACROSS\n|^Across\n|^ACROSS \n|^Across \n') ]))
corpusdf = corpusdf[~corpusdf.body.str.contains('^ACROSS\n|^Across\n|^ACROSS \n|^Across \n') ]


# %% write to processed data folder as this is not the final version
corpusdf.to_pickle(processeddatapath/'prepared_corpusdf.pickle')
corpusdf.to_csv(processeddatapath/'prepared_corpusdf.csv', index=False)

print("The total number of rows in the corpus is ",  corpusdf.shape[0])
# %%

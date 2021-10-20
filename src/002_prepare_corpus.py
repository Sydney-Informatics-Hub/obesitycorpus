import pandas as pd
from utils import get_project_root
from functs import get_byline, parse_filename, readfilesin, make_slug
from functs import get_text4digitcode, clean_nonascii, clean_quotes, get_date, clean_quot, replace_six_questionmarks
from functs import write_corpus_titlebody, write_corpus_sketchengine, get_wordcount_from_metadata, cqpweb_metadata, write_corpus_nested
from functs import clean_page_splits, clean_redundant_phrases, count_keywords, strip_newlines, apply_to_titlebody
from functs import clean_wa
from datetime import datetime
import re
import shutil


# imports for debugging non-unicode characters
# from functs import find_problems, find_filename_from_string, find_specific_character_with_preceding, 
# from functs import print_body_from_string, find_specific_character_wout_preceding, 
# from functs import where_is_byline

projectroot = str(get_project_root())
rawdatapath = projectroot + "/100_data_raw/Australian Obesity Corpus/"
processeddatapath = projectroot + "/300_data_processed/"

corpusdf = pd.read_csv(str(processeddatapath + 'inferred_dataset_encodings.csv'))

# get column with files as full path
corpusdf['fullpath'] = rawdatapath + corpusdf['filename'].astype(str)
print(corpusdf.shape[0])

result = [readfilesin(x, y) for x, y in zip(corpusdf['fullpath'], corpusdf['encoding'])]

corpusdf['contents'] = result

# filter out those files that don't have the word "body" (case-insensitive) in them
# These are the list files and the zip file
# write out the metadata for them for records
dropped = corpusdf[~corpusdf['contents'].str.contains('body', case=False)]
dropped.to_csv(f'../200_data_clean/dropped.csv', index=False)
# then filter them out
corpusdf = corpusdf[corpusdf['contents'].str.contains('body', case=False)]

# split header and body, thankfully case-sensitive works
corpusdf['contents_split'] = corpusdf['contents'].str.split("\nBody\n")
# always splits into two
# corpusdf['contents_len'] = corpusdf['contents_split'].str.len()
# take advantage of this:
corpusdf[['metadata', 'body']] = pd.DataFrame(corpusdf.contents_split.tolist(), index=corpusdf.index)
corpusdf = corpusdf.drop(columns=['contents_split', 'contents'])

# grab the best pass at the title, and convert all so only the first word is capitalised
corpusdf = corpusdf.assign(title=corpusdf['metadata'].map(lambda x: x.partition('\n')[0].capitalize()))

# get date/month from the relevant directory of the filepath
filename_fields = corpusdf["filename"].apply(parse_filename)
corpusdf = corpusdf.assign(**filename_fields)
# extract the four digit code from the filename
corpusdf = corpusdf.assign(fourdigitcode=corpusdf['filename'].map(get_text4digitcode))

# where are the bylines?
# code used for eda but not needed later
# corpusdf = corpusdf.assign(byline_locs = corpusdf['contents_split'].map(lambda x: where_is_byline(x)))
# corpusdf['bylines_flat'] = [','.join(map(str, l)) for l in corpusdf['byline_locs']]
corpusdf = corpusdf.assign(byline=corpusdf['metadata'].map(lambda x: get_byline(x)))
# lots of room to clean bylines if needed/desired...

# the below results in an empty set
# asciibodies = corpusdf[corpusdf.encoding == "ascii"].body.tolist()
# set([item for sublist in [re.findall(r'[^\x00-\x7F]+',x) for x in asciibodies]  for item in sublist])

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
# and replace multiple dashes with just one
corpusdf["body"] = corpusdf["body"].apply(lambda x: (re.sub(r'-+', '-', x)))
# and then replace ampersands with safe replacements
clean_amp = lambda x: (re.sub(r'&', '&amp;', x))
apply_to_titlebody(corpusdf, clean_amp)
# replace some odd Western Australia quote/text box, observed in the titles
apply_to_titlebody(corpusdf, clean_wa)

# get dates
corpusdf['date'] = corpusdf["metadata"].apply(get_date)
# some may not parse
print("The total number of files where we were unable to find a date in the metadata was ", sum(corpusdf.date.isna()), "files.")
print("These will be replaced from the filename, setting the date to the 1st of the month.")
# if unable to be detected replace with the first of the month and year that exists in the filename
# at least for this corpus - 1 file, no date in raw file
corpusdf['tmpdate'] = corpusdf.apply(lambda x: (datetime.strptime(str(x['year'] + "-" + x['numeric_month'] + "-01 00:00:00"), "%Y-%m-%d %H:%M:%S")), axis=1)
corpusdf['date'] = corpusdf['date'].fillna(corpusdf['tmpdate'])
corpusdf.drop('tmpdate', axis=1, inplace=True)

# clean page references and social media references
corpusdf["body"] = corpusdf["body"].apply(clean_page_splits)
corpusdf["body"] = corpusdf["body"].apply(clean_redundant_phrases)

# wordcount
# extract length from metadata
corpusdf = corpusdf.assign(wordcount_from_metatata=corpusdf['metadata'].map(lambda x: get_wordcount_from_metadata(x)))
# Get a basic word count of the body (note: without title)
corpusdf['wordcount_body'] = corpusdf['body'].str.count(' ') + 1
corpusdf['wordcount_title'] = corpusdf['title'].str.count(' ') + 1
corpusdf['wordcount_total'] = corpusdf.loc[:,['wordcount_body','wordcount_title']].sum(axis=1)

# can use the below functions to narrow down issues ---
# find_problems(start, end, colname = "cleaned_bodies", corpusdf = corpusdf)
# find_specific_character_with_preceding(character, start, end, colname = "cleaned_bodies", corpusdf = corpusdf)
# find_specific_character_wout_preceding(character, start, end, colname = "cleaned_bodies", corpusdf = corpusdf)
# find_filename_from_string(string, corpusdf = corpusdf)
# print_body_from_string(string, corpusdf = corpusdf)

# count how many times words of interest appear in the body
corpusdf['obesity_body_count'] = count_keywords(corpusdf['body'], 'obesity')
corpusdf['obesogen_body_count'] = count_keywords(corpusdf['body'], 'obesogen')
corpusdf['obese_body_count'] = count_keywords(corpusdf['body'], 'obese')
corpusdf['keywords_sum_body'] = corpusdf.obesity_body_count + corpusdf.obesogen_body_count + corpusdf.obese_body_count
#
corpusdf['obesity_in_title'] = count_keywords(corpusdf['title'], 'obesity')
corpusdf['obesogen_in_title'] = count_keywords(corpusdf['title'], 'obesogen')
corpusdf['obese_in_title'] = count_keywords(corpusdf['title'], 'obese')
corpusdf['keywords_sum_title'] = corpusdf.obesity_in_title + corpusdf.obesogen_in_title + corpusdf.obese_in_title
#
corpusdf['keywords_sum_total'] = corpusdf.keywords_sum_body + corpusdf.keywords_sum_title



# write corpus out as per client's request
write_corpus_titlebody(df=corpusdf, directoryname="corpus-titlebody")
cqpweb_metadata(df = corpusdf, directoryname="corpus-titlebody")
write_corpus_sketchengine(df=corpusdf, directoryname="corpus-sketchengine")
write_corpus_nested(df=corpusdf, directoryname="corpus-nested")

corpusdf.to_pickle("../200_data_clean/corpusdf.pickle")

# copy to cloudstor
# very fragile local link :(
shutil.copytree('../200_data_clean', '../../../../cloudstor/projects/obesity', dirs_exist_ok=True)
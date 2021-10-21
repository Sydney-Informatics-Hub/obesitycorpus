import pandas as pd
from utils import get_project_root
from functs import get_byline, parse_filename, readfilesin, make_slug
from functs import get_text4digitcode, clean_nonascii, clean_quotes, get_date, clean_quot, replace_six_questionmarks
from functs import write_corpus_titlebody, write_corpus_sketchengine, get_wordcount_from_metadata, cqpweb_metadata, write_corpus_nested
from functs import clean_page_splits, clean_redundant_phrases, count_keywords
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

filesdf = pd.read_csv(str(processeddatapath + 'inferred_dataset_encodings.csv'))

# get column with files as full path
filesdf['fullpath'] = rawdatapath + filesdf['filename'].astype(str)
print(filesdf.shape[0])

result = [readfilesin(x, y) for x, y in zip(filesdf['fullpath'], filesdf['encoding'])]

filesdf['contents'] = result

# filter out those files that don't have the word "body" (case-insensitive) in them
# These are the list files and the zip file
filesdf = filesdf[filesdf['contents'].str.contains('body', case=False)]

# split header and body, thankfully case-sensitive works
filesdf['contents_split'] = filesdf['contents'].str.split("\nBody\n")
# always splits into two
# filesdf['contents_len'] = filesdf['contents_split'].str.len()
# take advantage of this:
filesdf[['metadata', 'body']] = pd.DataFrame(filesdf.contents_split.tolist(), index=filesdf.index)
filesdf = filesdf.drop(columns=['contents_split', 'contents'])

# grab the best pass at the title, and convert all to title case
filesdf = filesdf.assign(title=filesdf['metadata'].map(lambda x: x.partition('\n')[0].title()))

# get date/month from the relevant directory of the filepath
filename_fields = filesdf["filename"].apply(parse_filename)
filesdf = filesdf.assign(**filename_fields)
# extract the four digit code from the filename
filesdf = filesdf.assign(fourdigitcode=filesdf['filename'].map(get_text4digitcode))

# where are the bylines?
# code used for eda but not needed later
# filesdf = filesdf.assign(byline_locs = filesdf['contents_split'].map(lambda x: where_is_byline(x)))
# filesdf['bylines_flat'] = [','.join(map(str, l)) for l in filesdf['byline_locs']]
filesdf = filesdf.assign(byline=filesdf['metadata'].map(lambda x: get_byline(x)))
# lots of room to clean bylines if needed/desired...

# the below results in an empty set
# asciibodies = filesdf[filesdf.encoding == "ascii"].body.tolist()
# set([item for sublist in [re.findall(r'[^\x00-\x7F]+',x) for x in asciibodies]  for item in sublist])

# clean up non-unicode characters
filesdf["body"] = filesdf["body"].apply(clean_nonascii)
filesdf["title"] = filesdf["title"].apply(clean_nonascii)
filesdf["metadata"] = filesdf["metadata"].apply(clean_nonascii)

# clean up quotes
filesdf["body"] = filesdf["body"].apply(clean_quotes)
filesdf["title"] = filesdf["title"].apply(clean_quotes)
# and the random repeats of "&quot;"
filesdf["title"] = filesdf["title"].apply(clean_quot)
filesdf["body"] = filesdf["body"].apply(clean_quot)
# and replace six ?????? specifically with double quotes...
filesdf["body"] = filesdf["body"].apply(replace_six_questionmarks)
# and replace multiple dashes with just one
filesdf["body"] = filesdf["body"].apply(lambda x: (re.sub(r'-+', '-', x)))


# get dates
filesdf['date'] = filesdf["metadata"].apply(get_date)
# some may not parse
print("The total number of files where we were unable to find a date in the metadata was ", sum(filesdf.date.isna()), "files.")
print("These will be replaced from the filename, setting the date to the 1st of the month.")
# if unable to be detected replace with the first of the month and year that exists in the filename
# at least for this corpus - 1 file, no date in raw file
tmpdate = filesdf.apply(lambda x: (datetime.strptime(str(x['year'] + "-" + x['numeric_month'] + "-01 00:00:00"), "%Y-%m-%d %H:%M:%S")), axis=1)
filesdf['date'] = filesdf['date'].fillna(tmpdate)

# clean page references and social media references
filesdf["body"] = filesdf["body"].apply(clean_page_splits)
filesdf["body"] = filesdf["body"].apply(clean_redundant_phrases)

# wordcount
# extract length from metadata
filesdf = filesdf.assign(wordcount_from_metatata=filesdf['metadata'].map(lambda x: get_wordcount_from_metadata(x)))
# Get a basic word count of the body (note: without title)
filesdf['wordcount_body'] = filesdf['body'].str.count(' ') + 1
filesdf['wordcount_title'] = filesdf['title'].str.count(' ') + 1
filesdf['wordcount_total'] = filesdf.loc[:,['wordcount_body','wordcount_title']].sum(axis=1)

# can use the below functions to narrow down issues ---
# find_problems(start, end, colname = "cleaned_bodies", filesdf = filesdf)
# find_specific_character_with_preceding(character, start, end, colname = "cleaned_bodies", filesdf = filesdf)
# find_specific_character_wout_preceding(character, start, end, colname = "cleaned_bodies", filesdf = filesdf)
# find_filename_from_string(string, filesdf = filesdf)
# print_body_from_string(string, filesdf = filesdf)

# count how many times words of interest appear in the body
filesdf['obesity_body_count'] = count_keywords(filesdf['body'], 'obesity')
filesdf['obesogen_body_count'] = count_keywords(filesdf['body'], 'obesogen')
filesdf['obese_body_count'] = count_keywords(filesdf['body'], 'obese')
filesdf['keywords_sum_body'] = filesdf.obesity_body_count + filesdf.obesogen_body_count + filesdf.obese_body_count
#
filesdf['obesity_in_title'] = count_keywords(filesdf['title'], 'obesity')
filesdf['obesogen_in_title'] = count_keywords(filesdf['title'], 'obesogen')
filesdf['obese_in_title'] = count_keywords(filesdf['title'], 'obese')
filesdf['keywords_sum_title'] = filesdf.obesity_in_title + filesdf.obesogen_in_title + filesdf.obese_in_title
#
filesdf['keywords_sum_total'] = filesdf.keywords_sum_body + filesdf.keywords_sum_title



# write corpus out as per client's request
write_corpus_titlebody(df=filesdf, directoryname="corpus-titlebody")
cqpweb_metadata(df = filesdf, directoryname="corpus-titlebody")
write_corpus_sketchengine(df=filesdf, directoryname="corpus-sketchengine")
write_corpus_nested(df=filesdf, directoryname="corpus-nested")

filesdf.to_pickle("../200_data_clean/filesdf.pickle")

# copy to cloudstor
# very fragile local link :(
shutil.copytree('../200_data_clean', '../../../../cloudstor/projects/obesity', dirs_exist_ok=True)

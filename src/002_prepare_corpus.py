import os
import pathlib
from collections import Counter
import pandas as pd
import re
from utils import get_project_root
import functs as f
from unidecode import unidecode

projectroot = str(get_project_root()) 
rawdatapath = projectroot + "/100_data_raw/Australian Obesity Corpus/"
processeddatapath = projectroot + "/300_data_processed/"

filesdf = pd.read_csv(str(processeddatapath + 'inferred_dataset_encodings.csv'))

# get column with files as full path
filesdf['fullpath'] = rawdatapath + filesdf['filename'].astype(str)
print(filesdf.shape[0])

result = [f.readfilesin(x, y) for x, y in zip(filesdf['fullpath'], filesdf['encoding'])]

filesdf['contents'] = result


# filter out those files that don't have the word "body" (case-insensitive) in them
# these are the list files and the zip file
filesdf = filesdf[filesdf['contents'].str.contains('body', case = False)]

# split header and body, thankfully case-sensitive works
filesdf['contents_split'] =  filesdf['contents'].str.split("\nBody\n")
# always splits into two
# filesdf['contents_len'] = filesdf['contents_split'].str.len()
# take advantage of this:
filesdf[['metadata','body']] = pd.DataFrame(filesdf.contents_split.tolist(), index= filesdf.index)
filesdf = filesdf.drop(columns=['contents_split', 'contents'])

# grab the best pass at the title, and convert all to title case
filesdf = filesdf.assign(title = filesdf['metadata'].map(lambda x: x.partition('\n')[0].title()))

# get date/month from the relevant directory of the filepath
filesdf = filesdf.assign(relevant_column = filesdf['filename'].map(lambda x: x.split('/')[2].split("_")))
filesdf = filesdf.assign(text_iterable = filesdf['filename'].map(lambda x: x.split('/')[3].replace("txt","")))

filesdf['source'] = filesdf['relevant_column'].str[0]
filesdf['year'] = filesdf['relevant_column'].str[1]
filesdf['month'] = filesdf['relevant_column'].str[2].str.replace("txt", "").str.strip()
# clean up the month labels
filesdf['numeric_month'] = filesdf['month'].map(lambda x: f.convert_month(x))

# where are the bylines?
# code used for eda but not needed later
#filesdf = filesdf.assign(byline_locs = filesdf['contents_split'].map(lambda x: f.where_is_byline(x)))
#filesdf['bylines_flat'] = [','.join(map(str, l)) for l in filesdf['byline_locs']]
filesdf = filesdf.assign(byline = filesdf['metadata'].map(lambda x: f.get_byline(x)))
# lots of room to clean bylines if needed/desired...

# clean up unicode in body
filesdf['cleaned_bodies']=filesdf.body


# the below results in an empty set
# asciibodies = filesdf[filesdf.encoding == "ascii"].body.tolist()
# set([item for sublist in [re.findall(r'[^\x00-\x7F]+',x) for x in asciibodies]  for item in sublist])

replacementdictionary = {"Â\xad": "' ", "~\xad" : "-", "\\xE2Ä(tm)":"'", "\\xE2Äú":"\"",\
     "\\xE2Ä\"": "-","\xE2Äò": "\"", "\\xE2€(tm)":"'", "\\xE2€": "'"}
replacementdictionary.update(pd.read_csv("replacements.csv",quotechar="'", escapechar="\\",\
    keep_default_na=False).set_index('word')['replacement'].to_dict())

filesdf = filesdf.assign(cleaned_bodies = filesdf['cleaned_bodies'].map(lambda x: f.mystringreplace(x, replacementdictionary)))

# can use the below functions to narrow down issues ---
# f.find_problems(start, end, colname = "cleaned_bodies")
# f.find_specific_character_with_preceding(character, start, end, colname = "cleaned_bodies")
# f.find_specific_character_wout_preceding(character, start, end, colname = "cleaned_bodies")
# f.find_filename_from_string(string)
# f.print_body_from_string(string)

# this needs to happen after the replacement dictionary replacement as it will
# automatically incorrectly replace all non-ascii characters
filesdf = filesdf.assign(cleaned_bodies = filesdf['cleaned_bodies'].map(lambda x: unidecode(x)))

# write corpus out as per client's request
filesdf['outputfilename'] = str("../200_data_clean/corpus/") +  filesdf['source'] + "_" + \
    filesdf['year'] + "_" + filesdf['numeric_month'] + "_" + \
        filesdf['text_iterable'] + "_" + \
        filesdf.title.apply(lambda x: f.cleantitler(x)) + ".txt"

for index, row in filesdf.iterrows():
    if index > len(filesdf):
       break
    else:
       f = open(row['outputfilename'], 'w', encoding='utf-8')
       f.write(row['body'])
       f.close()
       index+=1


filesdf.to_pickle("../200_data_clean/filesdf.pickle" )


#filesdf.to_csv("resonabledf.csv")











    




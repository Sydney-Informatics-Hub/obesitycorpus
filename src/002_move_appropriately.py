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
messybodies = filesdf[filesdf.encoding != "ascii"].body.tolist()


# this results in an empty set
# asciibodies = filesdf[filesdf.encoding == "ascii"].body.tolist()
# set([item for sublist in [re.findall(r'[^\x00-\x7F]+',x) for x in asciibodies]  for item in sublist])

#MOVE LATER
def mystringreplace(string, replacementobject):
    if string is None:
        return(None)
    elif isinstance(replacementobject, list):
        for word in replacementobject:
            string = string.replace(word, " ")
        return(string)
    else:
        for word, replacement in replacementobject.items():
            string = string.replace(word, replacement)
        return(string)




replacementdictionary = pd.read_csv("replacements.csv",quotechar="'", escapechar="\\").set_index('word')['replacement'].to_dict()
# the below breaks when loading from files
replacementdictionary["Â\xad"] = " "
replacementdictionary["~\xad"] = "-"

messybodies = [mystringreplace(x, replacementdictionary) for x in messybodies]

[item for sublist in [re.findall(r'\w+.[^\x00-\x7F].+',x) for x in messybodies[1100:1125]]  for item in sublist]

# this needs to happen after the replacement dictionary replacement as it will
# automatically incorrectly replace all non-ascii charachters
messybodies = [unidecode(x) for x in messybodies]



# filesdf.byline[19000:20100].apply(lambda x: mystringreplace(x, byline_replacementlist)).value_counts()

# [getweird(element, index) for index, element in enumerate(allbodies)]

# re.findall(r'[a-zA-Z]*[^\x00-\x7F][a-zA-Z]*',messybodies[1])
# re.findall(r'[^\x00-\x7F]+',messybodies[0])""
# set([item for sublist in [re.findall(r'[^\x00-\x7F]+',x) for x in messybodies]  for item in sublist])

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











    




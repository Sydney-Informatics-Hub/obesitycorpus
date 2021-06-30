import os
import pathlib
from collections import Counter
from chardet import detect
import pandas as pd
import re
from utils import get_project_root
import functs as f

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

# split header and body, thankfully case-senstive works
filesdf['contents_split'] =  filesdf['contents'].str.split("\nBody\n")
# always split into two
# filesdf['contents_len'] = filesdf['contents_split'].str.len()

# get date/month from the relevant directory of the filepath
filesdf = filesdf.assign(relevant_column = filesdf['filename'].map(lambda x: x.split('/')[2].split("_")))
filesdf['source'] = filesdf['relevant_column'].str[0]
filesdf['year'] = filesdf['relevant_column'].str[1]
filesdf['month'] = filesdf['relevant_column'].str[2].str.replace("txt", "").str.strip()
# clean up the month labels
filesdf['short_month'] = filesdf['month'].map(lambda x: f.convert_month(x))


# where are the bylines?
filesdf = filesdf.assign(byline_locs = filesdf['contents_split'].map(lambda x: f.where_is_byline(x)))



#filesdf.to_csv("resonabledf.csv")










    




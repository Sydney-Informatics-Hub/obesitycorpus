import os
import pathlib
from collections import Counter
from bs4 import UnicodeDammit
from chardet import detect
import pandas as pd
import re

# myfile = ['/Users/darya/DropboxSydneyUni/Projects/pipe-1951-obesity/data/raw/Australian Obesity Corpus/The Advertiser/Advertiser_2015/Advertiser_2015_Octobertxt/0007txt/Advertiser (1).txt']

filesdf = pd.read_csv('inferred_dataset_encodings.csv')


def readfilesin(file_path, encoding):
    if encoding in ['ascii', 'Windows-1252', 'ISO-8859-1']:
        with open(file_path, encoding='Windows-1252') as file:
            data = file.read().replace("\r", "")
    elif encoding == 'utf-8':
        with open(file_path, encoding='utf-8') as file:
            data = file.read().replace("\r", "")
    else:
        try:
            with open(file_path, 'rb') as non_unicode_file:
                content = non_unicode_file.read(1024)
                dammit = UnicodeDammit(content, ['Windows-1252'])
                data = dammit.unicode_markup.replace("\r", "")
        except Exception as e:
            raise ValueError('Can\'t return dictionary from empty or invalid file %s due to %s' % (file_path, e))
    return(data)


print(filesdf.shape[0])

result = [readfilesin(x, y) for x, y in zip(filesdf['Filename'], filesdf['Encoding'])]

filesdf['Contents'] = result
filesdf['Contents_split'] =  filesdf['Contents'].str.split("\n\n\n")
filesdf['Contents_len'] = filesdf['Contents_split'].str.len()

# filter out those that are > 10 chunks as they are lists of texts (search results)
# instead of the actual texts themselves
reasonabledf = filesdf[filesdf['Contents_len'] < 10]

# get date/month
reasonabledf['year'] = reasonabledf['Filename'].



# reasonabledf.to_csv("resonabledf.csv")

# '/Users/darya/DropboxSydneyUni/Projects/pipe-1951-obesity/data/raw/Australian Obesity Corpus/The Advertiser/Advertiser_2009/Advertiser_2009_Octobertxt/0018txt/Advertiser (1).txt'
reasonabledf['Source'] = reasonabledf['Filename'].str.split('/').str[11].str.split("_").str[0]




#smaller = filesdf[filesdf["Contents_len"] > 10] 
#print(smaller['Filename'].tolist())
#print(smaller['Contents_len'].value_counts())


    




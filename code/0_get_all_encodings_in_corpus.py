import os
import pathlib
import chardet
import pandas as pd
from pathlib import Path
# this is my personal file with common variables
# from commonvars import path
path = "/Users/darya/DropboxSydneyUni/Projects/pipe-1951-obesity/data/raw/Australian Obesity Corpus/"



listoffiles = []

for path, subdirs, files in os.walk(path):
    [listoffiles.append(str(pathlib.PurePath(path, name))) for name in files]

# make sure we're only looking at the txt files and not zips etc
excludedfiles = [filename for filename in listoffiles if filename[-3:] != "txt"]
for i in excludedfiles:
    print("We have excluded the following file from analysis (wrong extension)", i)

listoffiles = [filename for filename in listoffiles if filename[-3:] == "txt"]

encodingtypes = []
confidence = []
for file in listoffiles:
    with open(file, 'rb') as detect_file_encoding:
        detection = chardet.detect(detect_file_encoding.read())
        encodingtypes.append(detection['encoding'])
        confidence.append(detection['confidence'])

dataformats = pd.DataFrame(list(zip(listoffiles, encodingtypes,confidence)),
               columns =['Filename', 'Encoding', 'Confidence'])
dataformats.to_csv("inferred_dataset_encodings.csv")
print(dataformats['Encoding'].value_counts())
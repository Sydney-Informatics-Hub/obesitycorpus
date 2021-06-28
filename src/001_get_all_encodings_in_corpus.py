import os
import pathlib
import chardet
import pandas as pd
from utils import get_project_root

projectroot = str(get_project_root()) 
datapath = projectroot + "/100_data_raw/Australian Obesity Corpus/"


listoffiles = []

for path, subdirs, files in os.walk(datapath):
    [listoffiles.append(str(pathlib.PurePath(path, name).relative_to(datapath))) for name in files]

# make sure we're only looking at the txt files and not zips etc
excludedfiles = [filename for filename in listoffiles if filename[-3:] != "txt"]
for i in excludedfiles:
    print("We have excluded the following file from analysis (wrong extension)", i)

listoffiles = [filename for filename in listoffiles if filename[-3:] == "txt"]



encodingtypes = []
confidence = []
for file in listoffiles:
    with open(str(datapath + file), 'rb') as detect_file_encoding:
        detection = chardet.detect(detect_file_encoding.read())
        encodingtypes.append(detection['encoding'])
        confidence.append(detection['confidence'])

dataformats = pd.DataFrame(list(zip(listoffiles, encodingtypes,confidence)),
               columns =['filename', 'encoding', 'confidence'])

dataformats.to_csv(str(projectroot + "/300_data_processed/" +  "inferred_dataset_encodings.csv"))
print(dataformats['encoding'].value_counts())
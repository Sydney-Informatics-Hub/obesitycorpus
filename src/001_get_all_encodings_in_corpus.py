# %%
import os
import pathlib
import chardet
import pandas as pd
import pathlib
from utils import get_projectpaths
(projectroot, rawdatapath, cleandatapath, processeddatapath) = get_projectpaths()

# %%
listoffiles = []

for path, subdirs, files in os.walk(rawdatapath):
    [listoffiles.append(str(pathlib.PurePath(path, name).relative_to(rawdatapath))) for name in files]

# %% make sure we're only looking at the txt files and not zips etc
excludedfiles = [filename for filename in listoffiles if filename[-3:] != "txt"]
for i in excludedfiles:
    print("We have excluded the following file from analysis (wrong extension)", i)
listoffiles = [filename for filename in listoffiles if filename[-3:] == "txt"]


# %% get encodings
encodingtypes = []
confidence = []
for file in listoffiles:
    with open(str(rawdatapath/ file), 'rb') as detect_file_encoding:
        detection = chardet.detect(detect_file_encoding.read())
        encodingtypes.append(detection['encoding'])
        confidence.append(detection['confidence'])

# %% get data formats
dataformats = pd.DataFrame(list(zip(listoffiles, encodingtypes,confidence)),
               columns =['filename', 'encoding', 'confidence'])

# %% export and count
dataformats.to_csv(processeddatapath/"inferred_dataset_encodings.csv", index=False)
print(dataformats['encoding'].value_counts())
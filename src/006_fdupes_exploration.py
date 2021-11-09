import pandas as pd
import pathlib
from utils import get_project_root

projectroot = str(get_project_root())
cleandatapath = pathlib.Path(projectroot +  "/200_data_clean")

with open(str(cleandatapath/'duplicatesinfiles.txt')) as f:
    i=0
    mylist = []
    numlist = []
    for line in f:
        if line[0]==".":
            mylist.append(line.lstrip("./").rstrip(".txt\n"))
            numlist.append(i)
        else:
            # blank line, need to start next batch
            i+=1

fdupes = pd.DataFrame(mylist, columns =['article'])
fdupes['duplicate_id'] = numlist

fdupes[['source', 'year', 'numeric_month', 'four_digit_code', 'title']] = fdupes['article'].str.split('_',expand=True)

# find duplicates by id
tmp = fdupes.filter(['source', 'duplicate_id'])
count_dupes = pd.crosstab(tmp["duplicate_id"], tmp["source"])

(count_dupes
 .groupby(count_dupes.columns.to_list())
 .size()
 .reset_index(name='Count')
 .sort_values('Count', ascending=False)
 .to_csv(str(str(cleandatapath) +"/identical_duplicates_fdupes.csv"), index=False))
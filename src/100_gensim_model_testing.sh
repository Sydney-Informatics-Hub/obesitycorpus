#! /bin/bash
cd /home/dvanichkina/scratch_sih/obesity
module purge
eval "$(~/bin/micromamba shell hook -s posix)"
micromamba activate obesity
python gensim_model_testing.py $NTOPICS
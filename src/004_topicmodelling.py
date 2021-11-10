# %% [markdown]
# ## Topic modelling of the corpus

# %%
import pathlib
from utils import get_projectpaths
(projectroot, rawdatapath, cleandatapath, processeddatapath) = get_projectpaths()

import re
import numpy as np
import pandas as pd
# silence annoying warning
pd.options.mode.chained_assignment = None  # default='warn'

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy
import matplotlib.pyplot as plt
from pprint import pprint
#%matplotlib inline

# Plotting tools
import pyLDAvis
# may need more from here https://stackoverflow.com/questions/66759852/no-module-named-pyldavis
import pyLDAvis.gensim_models as gensimvis

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

#
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# %%
# load data and obesity names
corpusdf = pd.read_pickle(str(cleandatapath/"corpusdf.pickle"))
# drop unneccessary columns
corpusdf = corpusdf.drop(['filename', 'encoding', 'confidence', 'fullpath','year', 'original_numeric_month'], axis=1)



# %%
# Convert body to list
bodies = corpusdf.body.values.tolist()

# Remove new line characters
bodies = [re.sub('\s+', ' ', sent) for sent in bodies]

# Remove single quotes
bodies = [re.sub("\'", "", sent) for sent in bodies]

# Remove double quotes
bodies = [re.sub('"', "", sent) for sent in bodies]

# %%
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

bodies_words = list(sent_to_words(bodies))

# %%
# Build the bigram and trigram models
bigram = gensim.models.Phrases(bodies_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[bodies_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# %%
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# %%
# Remove Stop Words
bodies_words_nostops = remove_stopwords(bodies_words)

# Form Bigrams
bodies_words_bigrams = make_bigrams(bodies_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
bodies_lemmatized = lemmatization(bodies_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# %%
# Create Dictionary
id2word = corpora.Dictionary(bodies_lemmatized)

# Create Corpus
texts = bodies_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# %%
# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=20,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

# %%
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# %%
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=bodies_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# %%
import pyLDAvis.gensim_models as gensim_models
# Visualize the topics
pyLDAvis.enable_notebook()
vis = gensim_models.prepare(lda_model, corpus, id2word)
vis

# %% [markdown]
# ## Key topics:
# ### Drinks
# - (0,
#   '0.173*"sugar" + 0.080*"drink" + 0.038*"water" + 0.031*"soft_drink" + '
#   '0.025*"add" + 0.019*"milk" + 0.019*"chocolate" + 0.018*"alcohol" + '
#   '0.017*"contain" + 0.017*"consumption"')
# 
# ### Close to topic about food
# 
# - (1,
#   '0.091*"food" + 0.062*"eat" + 0.030*"diet" + 0.027*"healthy" + 0.024*"say" + '
#   '0.019*"fat" + 0.017*"fruit" + 0.016*"vegetable" + 0.012*"meal" + '
#   '0.010*"make"'),
# 
# ### Also some overlap - pets and fertility?
# - (2,
#   '0.114*"dog" + 0.062*"animal" + 0.039*"pet" + 0.026*"user" + 0.025*"vic" + '
#   '0.018*"human" + 0.018*"illegal" + 0.016*"fertility" + 0.014*"leadership" + '
#   '0.014*"owner"'),
# 
# ### Exercise
#  - (3,
#   '0.030*"day" + 0.030*"exercise" + 0.023*"work" + 0.021*"time" + 0.019*"walk" '
#   '+ 0.019*"hour" + 0.015*"minute" + 0.015*"run" + 0.012*"week" + '
#   '0.011*"help"'),
# 
# - (4,
#   '0.032*"disease" + 0.030*"risk" + 0.027*"cancer" + 0.024*"heart" + '
#   '0.017*"high" + 0.015*"diabete" + 0.015*"level" + 0.014*"cause" + '
#   '0.013*"reduce" + 0.012*"also"'),
# 
# - (5,
#   '0.038*"government" + 0.023*"say" + 0.019*"obesity" + 0.019*"health" + '
#   '0.017*"australia" + 0.015*"cost" + 0.013*"tax" + 0.012*"australian" + '
#   '0.012*"state" + 0.012*"public"'),
# 
# - (6,
#   '0.011*"people" + 0.010*"many" + 0.010*"make" + 0.008*"well" + 0.008*"time" '
#   '+ 0.008*"take" + 0.007*"world" + 0.007*"use" + 0.006*"good" + 0.006*"give"'),
# 
# - (7,
#   '0.200*"food" + 0.081*"junk" + 0.080*"ban" + 0.062*"advertising" + '
#   '0.055*"fast" + 0.029*"unhealthy" + 0.025*"mcdonald" + 0.024*"outlet" + '
#   '0.019*"obesity" + 0.016*"marketing"'),
# 
# - (8,
#   '0.057*"health" + 0.041*"say" + 0.027*"hospital" + 0.025*"patient" + '
#   '0.019*"medical" + 0.017*"doctor" + 0.015*"surgery" + 0.013*"care" + '
#   '0.013*"dr" + 0.013*"need"'),
# 
# ### Women and money, some overlap with exercise
#  - (9,
#   '0.045*"cent" + 0.030*"say" + 0.025*"health" + 0.024*"year" + '
#   '0.023*"australian" + 0.018*"woman" + 0.017*"people" + 0.016*"rate" + '
#   '0.015*"high" + 0.015*"tasmanian"'),
# 
# - (10,
#   '0.116*"service" + 0.104*"mr" + 0.076*"community" + 0.034*"education" + '
#   '0.022*"ms" + 0.017*"act" + 0.016*"teacher" + 0.015*"mrs" + 0.012*"john" + '
#   '0.011*"significant"'),
# 
# - (11,
#   '0.035*"sport" + 0.031*"say" + 0.022*"player" + 0.022*"club" + 0.020*"afl" + '
#   '0.017*"labor" + 0.016*"last" + 0.016*"drug" + 0.014*"election" + '
#   '0.013*"year"'),
# 
# - (12,
#   '0.038*"cook" + 0.019*"oliver" + 0.018*"restaurant" + 0.018*"china" + '
#   '0.018*"christmas" + 0.015*"ambulance" + 0.015*"chef" + 0.015*"seat" + '
#   '0.013*"world" + 0.012*"country"'),
# 
# - (13,
#   '0.157*"pm" + 0.082*"police" + 0.026*"george" + 0.026*"wilkinson" + '
#   '0.022*"ministry" + 0.022*"jail" + 0.019*"hunt" + 0.015*"martin" + '
#   '0.015*"saturday" + 0.014*"friday"'),
# 
# - (14,
#   '0.037*"say" + 0.032*"get" + 0.027*"do" + 0.024*"go" + 0.022*"s" + '
#   '0.014*"think" + 0.014*"people" + 0.014*"make" + 0.012*"thing" + '
#   '0.012*"want"'),
# 
# - (15,
#   '0.017*"show" + 0.014*"year" + 0.013*"man" + 0.009*"first" + 0.009*"woman" + '
#   '0.009*"love" + 0.008*"new" + 0.007*"old" + 0.006*"tv" + 0.006*"star"'),
# 
# - (16,
#   '0.113*"child" + 0.063*"school" + 0.042*"parent" + 0.037*"say" + 0.034*"kid" '
#   '+ 0.017*"student" + 0.015*"family" + 0.014*"young" + 0.013*"obesity" + '
#   '0.013*"childhood"'),
# 
# ## Separate judgemental articles?
# - (17,
#   '0.189*"weight" + 0.058*"fat" + 0.050*"lose" + 0.046*"overweight" + '
#   '0.045*"obese" + 0.044*"loss" + 0.039*"body" + 0.027*"obesity" + 0.024*"kg" '
#   '+ 0.024*"size"'),
# 
# ## Small topic - reports on research
# - (18,
#   '0.078*"study" + 0.058*"research" + 0.043*"find" + 0.036*"university" + '
#   '0.036*"say" + 0.033*"researcher" + 0.020*"professor" + 0.019*"dr" + '
#   '0.017*"brain" + 0.016*"new"'),
# 
# ## Weird - transport in Tasmania similar to research
# - (19,
#   '0.028*"city" + 0.027*"tasmania" + 0.019*"car" + 0.017*"road" + '
#   '0.016*"state" + 0.015*"transport

# %%
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                          id2word=id2word,
                          num_topics=num_topics,
                          random_state=100,
                          update_every=1,
                          chunksize=100,
                          passes=10,
                          alpha='auto',
                          per_word_topics=True)

        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# %%
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=bodies_lemmatized, start=10, limit=20, step=1)

# %%
# Show graph
limit=20; start=10; step=1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()



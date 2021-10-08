import textstat
from flair.embeddings import StackedEmbeddings,TransformerWordEmbeddings,ELMoEmbeddings
from transformers import pipeline
from GLTR import LM
from ratelimit import sleep_and_retry,limits
import nltk
import numpy as np
from tqdm.notebook import tqdm
from flair.data import Sentence
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy import spatial
from json import JSONDecodeError

def get_sentiment(m,sentiment_analysis):
    """
        Compute the sentiment class and confidence score for the given sentence
        Args:
            m: (string) sentence to be processed
    """
    result = sentiment_analysis(m)[0]
    label=(0 if result['label']=='NEGATIVE' else 1)
    score=result['score']
    return label,score
def get_textstat_features(m):
  """
    Compute different features from the text stat library, mostly contains text readability information 
      Args:
          m: (string) sentence to be processed
  """
  return textstat.flesch_reading_ease(m), \
          textstat.automated_readability_index(m), \
          # textstat.reading_time(m, ms_per_char=14.69), \
          # textstat.syllable_count(m),\


def glec_similarity(t,v,model):
  """
    Computes GLEC embedding model similarity between tenor and vehicle
  """
  try:
    sim=model.similarity(t,v)
    return sim
  except KeyError:
    return 0

def glec_sentence_stats(m):
  """
    Computes GLEC sentence embedding stats
  """
  words_embds=[]

  for w in m.split(" "):
    try:
      words_embds.append(model[w])
    except:
      print("word: ",w,"not found in glec")
      words_embds.append(np.zeros(model.vector_size))
  n_components=3
  if len(words_embds)<3:
    n_components=2
  pca=PCA(n_components=n_components)
  embeddings=pca.fit_transform(words_embds)
  cs=cosine_similarity(embeddings)
  cs=cs *(1-np.eye(cs.shape[0]))
  return cs.mean(),cs.var(),pca.explained_variance_.max()

def transformer_flair_sentence_stats(m,transformer_stacked_embeddings):
  """
    Computes Flair transformer embeddings sentence stats
  """
  words_embds=[]
  sentence = Sentence(m)
  transformer_stacked_embeddings.embed(sentence)
  for w in sentence:
    words_embds.append(w.embedding.detach().cpu().numpy())
  n_components=3
  if len(words_embds)<3:
    n_components=2
  pca=PCA(n_components=n_components)
  embeddings=pca.fit_transform(words_embds)
  cs=cosine_similarity(embeddings)
  cs=cs *(1-np.eye(cs.shape[0]))
  return cs.mean(),cs.var(),pca.explained_variance_.sum()

def gltr_stats(raw_text,lm):
  """
    Calculates the mean and variance of words rank from gpt2 generation standpoint
    Args:
        raw_text (string): sentence to be analyzed
  """
  payload = lm.check_probabilities(raw_text, topk=50)
  real_topK = payload["real_topk"]
  ranks = [i[0] for i in real_topK]
  preds = [i[1] for i in real_topK]
  return np.mean(preds),np.var(preds),np.max(ranks),np.abs(np.prod(preds))


@sleep_and_retry
@limits(calls=60, period=60)
def get_tenor_vehicle_relation(t,v):
  """
    calculates the relatedness between the tenor and vehicle from conceptnet
    args:
      t (string): tenor string
      v (string): vehicle string
  """
  base_url='http://api.conceptnet.io'
  query='/relatedness?node1=/c/en/{fw}&node2=/c/en/{sw}'.format(fw=t,sw=v)
  try:
    req=requests.get(base_url+query)
    obj=req.json()
    value=obj['value']
  except JSONDecodeError:
    print(t,v,req.status_code)
    return 0
  return value

def get_conceptnet_relation_matrix_stats(raw_text,conceptnet):
  """
    calculates the mean and variance of inter-words relatedness in a metaphor
    args:
        raw_text (string): sentence to be analyzed
  """
  words_embds=[]

  for w in raw_text.split(" "):
    if w=='-':
      continue
    try:
      words_embds.append(conceptnet[w.lower()])
    except:
      try:
        words_embds.append(conceptnet[lemm.lemmatize(w.lower())])
      except:
        print("Word ",w,"not found",raw_text)
        words_embds.append(np.zeros(300))
  n_components=3
  if len(words_embds)<3:
    n_components=2
  pca=PCA(n_components=n_components)
  embeddings=pca.fit_transform(words_embds)
  cs=cosine_similarity(embeddings)
  cs=cs *(1-np.eye(cs.shape[0]))
  return cs.mean(),cs.var(),pca.explained_variance_.max()

def get_embeddings_similarity(t,v,conceptnet):
  """
    calculates conceptnet embeddings similarity between tenor and vehicle
    args:
        t (string): tenor string
        v (string): vehicle string
  """ 
  try:
    w1=conceptnet[t]
    w2=conceptnet[v]
    return  1 - spatial.distance.cosine(w1, w2)
  except:
    print(t,v,"n ot in vocab")
    return 0
def get_wordnet_similarity(t,v):
  """
    calculates path similarity between 2 words using wordnet
    This function assumes both words are nouns
    args:
        t (string): tenor string
        v (string): vehicle string
  """ 
  # t=stemmer.stem(t)
  # v=stemmer.stem(v)
  try:
    cb = wn.synset('{t}.n.01'.format(t=t))
    ib = wn.synset('{v}.n.01'.format(v=v))
  except:
    # print(t,v, 'unable to create')
    return 0
  return cb.wup_similarity(ib)


def featurize(df):
  """
      Creates the featurized representation from the metaphor and tenor, vehicle pairs using the different methods provided aboce

  """
  words_embds={}
  print("#### Loading Conceptnet Embeddings #####")
  with open('./numberbatch-en-19.08.txt') as f:
    print(f.readline())
    for line in tqdm(f.readlines()):
        line=line.strip().split(" ")
        words_embds[line[0]]=np.array([float(d) for d in line[1:]])
    f.close()
  print("#### Loading Flair Transformer Embeddings #####")
  
  embedding_gpt = TransformerWordEmbeddings("./gpt2_medium_glec/")
  embedding_roberta= TransformerWordEmbeddings("roberta-large-openai-detector")
  transformer_stacked_embeddings=StackedEmbeddings([embedding_roberta,embedding_gpt])

  print("#### GLTR GLTR Language Model  #####")


  lm=LM(model_name_or_path='./gpt2_medium_glec/')
  conceptnet=words_embds
  lemm=nltk.WordNetLemmatizer()
  print("#### Loading Sentiment Analysis Pipeline #####")

  sentiment_analysis = pipeline("sentiment-analysis")


  print("#### Featurizing ... #####")
  features=[]
  for m,pm,t,v in tqdm(zip(df.M,df.processed_metaphor,df.tenor,df.vehicle)):
    t=lemm.lemmatize(t.lower())
    v=lemm.lemmatize(v.lower())
    features.append([*gltr_stats(pm,lm),\
    *transformer_flair_sentence_stats(pm,transformer_stacked_embeddings),\
    get_embeddings_similarity(t,v,conceptnet),\
    *get_sentiment(pm,sentiment_analysis),\
    *get_textstat_features(pm) ])
  return features

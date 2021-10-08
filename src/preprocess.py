import pandas as pd
import numpy as np
import regex as re
import spacy
import en_core_web_lg


def preprocess_column(raw_texts):
  """
    Preprocess the metaphor column by preprocessing each metaphor and returning
    the resulting array
    Args:
        raw_texts: (list(string)) the metaphors to be processed
  """
  nlp = en_core_web_lg.load()
  return [preprocess_text(s,nlp) for s in raw_texts]

def preprocess_text(raw_text,nlp):
  """
      Preprocesses the raw metaphor by removing sotp words and lemmatizing the words
      Args:
          raw_text: (string) the original metaphor text to be processed
          nlp: (spacy language object)
  """
  
  tokens=[]
  for token in nlp(raw_text):
    if not token.is_stop:
      tokens.append(token.lemma_)
  return " ".join(tokens)

def trim_text(t):
  """
      finds the beginning and ending of a number in the original string and slices out the number
      Args:
        t: (string) token to be processed
  """

  if len(t)<1:
    return t
  start_index=0
  end_index=len(t)-1
  while not t[start_index].isnumeric() and start_index<len(t):
    start_index+=1
  while not t[end_index].isnumeric()and end_index>-1:
    end_index-=1
  return t[start_index:end_index+1].replace(",",'.')

def read_text(file):
  """
      Read the katz dataset from the txt file provided
      Args:
          file: (string) file path
  """

  columns=["M","tenor","vehicle","label","CMP","ESI","MET","MGD","SRL","MIM","IMS","IMP","FAM","ALT"]
  columns_dict={k:[]for k in columns}
  lengths=[]
  with open(file,'r') as f:
    f.readline()
    for line in f.readlines():
      line_values=line.split(",")
      line_values_numbers=[trim_text(s) for s in ",".join(line_values[4:]).split("\"") if s != ',' and s !='\n']
      # print(line_values_numbers)
      matches=re.findall('^\d,|\"\d+,\d+\"|,\d,|,\d$', ",".join(line_values[4:]),overlapped=True)
      if len(matches)!=10:
        print(line)
        continue
      
      lengths.append(len(matches))
      line_values=line_values[:4]+[trim_text(s) for s in matches]
      for j in range(len(columns)):
        if j>2:
          columns_dict[columns[j]].append(float(line_values[j]))
        else:
          columns_dict[columns[j]].append(line_values[j])
  f.close()
  return pd.DataFrame(columns_dict)

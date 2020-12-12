from transformers import MarianTokenizer, MarianMTModel
from typing import List
from torch import cuda
from nltk.translate import bleu_score
from tqdm import tqdm


#Test az-en default model on tatoeba
src = 'az'
trgt = 'en'
az_en_mname = f'Helsinki-NLP/opus-mt-{src}-{trgt}'
az_en_model = MarianMTModel.from_pretrained(az_en_mname).to('cuda')
az_en_tokenizer = MarianTokenizer.from_pretrained(az_en_mname)
az_en_test_data_az_path="../data/az-en/az-en-tatoeba/Tatoeba.az-en.az"
az_en_test_data_en_path="../data/az-en/az-en-tatoeba/Tatoeba.az-en.en"
az_en_test_az_lines = open(az_en_test_data_az_path, "r").readlines()
az_en_test_en_lines = open(az_en_test_data_en_path, "r").readlines()

#The tokenizer, model, and test corpus
#Change this and uncomment the corresponding variables above to run different tests
tokenizer = az_en_tokenizer
model = az_en_model
src_text = az_en_test_az_lines
trgt_text = az_en_test_en_lines

#Batch size and number of batches
batch_size = 100
num_of_batches = 10

#Storage lists for Bleu scores 1-4
n1_scores = []
n2_scores = []
n3_scores = []
n4_scores = []

def batch_and_test(offset):
  #batch
  batch = tokenizer.prepare_seq2seq_batch(src_texts=src_text[offset:offset+batch_size], return_tensors="pt").to('cuda')
  #generate
  gen = model.generate(**batch).to('cuda')
  #decode
  words: List[str] = tokenizer.batch_decode(gen, skip_special_tokens=True)
  #calculate Bleu
  #NOTE: Bleu scores of a certain n only include sentences of that n or above for
  #  whichever is shorter, ref or trgt.
  #Example: the sentence ref="Hello there, my friend" trgt="Hello there, friend" would 
  #  only be represented in Bleu1-3 scores, not Bleu4 
  for i in range(0,batch_size):
    ref = trgt_text[offset+i].split()
    trgt = words[i].split()
    n1_scores.append(bleu_score.sentence_bleu([ref], trgt, weights=(1,0,0,0)))
    if len(ref)>1 and len(trgt)>1:
      n2_scores.append(bleu_score.sentence_bleu([ref], trgt, weights=(0.5,0.5,0,0)))
    if len(ref)>2 and len(trgt)>2:
      n3_scores.append(bleu_score.sentence_bleu([ref], trgt, weights=(0.33,0.33,0.33,0)))
    if len(ref)>3 and len(trgt)>3:
      n4_scores.append(bleu_score.sentence_bleu([ref], trgt, weights=(0.25,0.25,0.25,0.25)))
  #Clear your cache
  del batch
  del gen
  del words
  cuda.empty_cache()

for i in tqdm(range(0,num_of_batches)):
  batch_and_test(i*batch_size)

print(f"\nAverage Bleu scores across {batch_size*num_of_batches} examples")
print(f"Avg Bleu1 Score: {sum(n1_scores)/len(n1_scores)} using {len(n1_scores)} examples")
print(f"Avg Bleu2 Score: {sum(n2_scores)/len(n2_scores)} using {len(n2_scores)} examples")
print(f"Avg Bleu3 Score: {sum(n3_scores)/len(n3_scores)} using {len(n3_scores)} examples")
print(f"Avg Bleu4 Score: {sum(n4_scores)/len(n4_scores)} using {len(n4_scores)} examples")




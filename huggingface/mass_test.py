from transformers import MarianTokenizer, MarianMTModel
from typing import List
from torch import cuda
from nltk.translate import bleu_score
from tqdm import tqdm

n1_scores = []
n2_scores = []
n3_scores = []
n4_scores = []

#Batch size and number of batches
num_of_test_examples = 1000
batch_size = 50
assert num_of_test_examples%batch_size == 0, "Number of test examples not cleanly divided by batch size"
num_of_batches = int(num_of_test_examples/batch_size)

def batch_and_test(model, tokenizer, src_text, trgt_text, offset):
  global n1_scores
  global n2_scores
  global n3_scores
  global n4_scores
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

def write_data():
  global n1_scores
  global n2_scores
  global n3_scores
  global n4_scores
  output_file.write(f"Avg Bleu1 Score: {sum(n1_scores)/len(n1_scores)} using {len(n1_scores)} examples\n")
  output_file.write(f"Avg Bleu2 Score: {sum(n2_scores)/len(n2_scores)} using {len(n2_scores)} examples\n")
  output_file.write(f"Avg Bleu3 Score: {sum(n3_scores)/len(n3_scores)} using {len(n3_scores)} examples\n")
  output_file.write(f"Avg Bleu4 Score: {sum(n4_scores)/len(n4_scores)} using {len(n4_scores)} examples\n\n")
  n1_scores = []
  n2_scores = []
  n3_scores = []
  n4_scores = []

if __name__ == "__main__":
  num_of_models = 10
  model_dir = "./fine-tuned-model/"
  tatoeba_src = open("../data/az-en/az-en-tatoeba/Tatoeba.az-en.az", "r").readlines()
  tatoeba_trgt = open("../data/az-en/az-en-tatoeba/Tatoeba.az-en.en", "r").readlines()
  back_trans_src = open("./fine-tune-data/test.source", "r").readlines()
  back_trans_trgt = open("./fine-tune-data/test.target", "r").readlines()
  
  output_file = open("./output.txt", "w")
  
  tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-az-en")
  
  for i in range(1,num_of_models+1):
    print(f"Testing model{i}")
    output_file.write(f"model{i}\n")
    mname = f"./fine-tuned-model/model{i}/"
    model = MarianMTModel.from_pretrained(mname).to('cuda')
    for j in tqdm(range(0, num_of_batches)):
      batch_and_test(model, tokenizer, tatoeba_src, tatoeba_trgt, j*batch_size)
    output_file.write(f"tatoeba test set\n")
    write_data()
    for j in tqdm(range(0, num_of_batches)):
      batch_and_test(model, tokenizer, back_trans_src, back_trans_trgt, j*batch_size)
    output_file.write(f"back translated test set\n")
    write_data()
    output_file.write("\n")




    

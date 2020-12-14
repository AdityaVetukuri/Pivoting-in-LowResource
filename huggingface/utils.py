from transformers import MarianTokenizer, MarianMTModel
from typing import List
from torch import cuda as c

def translate_batch(model, tokenizer, src_text_list, cuda=True):
  if cuda:
    batch = tokenizer.prepare_seq2seq_batch(src_texts=src_text, return_tensors="pt").to('cuda')
    gen = model.generate(**batch).to('cuda')
    words: List[str] = tokenizer.batch_decode(gen, skip_special_tokens=True)
    c.empty_cache()
  else:
    batch = tokenizer.prepare_seq2seq_batch(src_texts=src_text, return_tensors="pt")
    gen = model.generate(**batch)
    words: List[str] = tokenizer.batch_decode(gen, skip_special_tokens=True)
  return words

if __name__ == "__main__":
  
  src = "az"
  trgt = "en"
  model_name = f"Helsinki-NLP/opus-mt-{src}-{trgt}"
  tokenizer = MarianTokenizer.from_pretrained(model_name)
  model = MarianMTModel.from_pretrained(model_name).to('cuda')
  
  src_text = ["Burun qandır."]
  words = translate_batch(model, tokenizer, src_text)
  
  
  
  for i in range(0,len(src_text)):
    print(src_text[i])
    print(words[i])
    print("")
  

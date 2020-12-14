import random
from tqdm import tqdm

#Input text
src_text_path = "./back-translate-data/back-trans-open-sub.az"
src_text_file = open(src_text_path, "r")
src_text = src_text_file.readlines()
src_text_file.close()
trgt_text_path = "./back-translate-data/back-trans-open-sub.en"
trgt_text_file = open(trgt_text_path, "r")
trgt_text = trgt_text_file.readlines()
trgt_text_file.close()

#Parameters
train_size = 100000
dev_size = 50000
test_size = 50000

#Shuffle
parallel_text = list(zip(src_text, trgt_text))
random.shuffle(parallel_text)
src_text, trgt_text = zip(*parallel_text)

#Write train set
src_file_out_path = "./fine-tune-data/train.source"
src_file_out = open(src_file_out_path,"w")
trgt_file_out_path = "./fine-tune-data/train.target"
trgt_file_out = open(trgt_file_out_path, "w")
print(f"Writing train set sized: {train_size}")
start = 0
stop = train_size
for i in tqdm(range(start,stop)):
  src_file_out.write(src_text[i])
  trgt_file_out.write(trgt_text[i])

#Write dev set
src_file_out_path = "./fine-tune-data/dev.source"
src_file_out = open(src_file_out_path,"w")
trgt_file_out_path = "./fine-tune-data/dev.target"
trgt_file_out = open(trgt_file_out_path, "w")
print(f"Writing dev set sized: {dev_size}")
start = train_size
stop = train_size + dev_size
for i in tqdm(range(start,stop)):
  src_file_out.write(src_text[i])
  trgt_file_out.write(trgt_text[i])

#Write test set
src_file_out_path = "./fine-tune-data/test.source"
src_file_out = open(src_file_out_path,"w")
trgt_file_out_path = "./fine-tune-data/test.target"
trgt_file_out = open(trgt_file_out_path, "w")
print(f"Writing test set sized: {test_size}")
start = train_size+dev_size
stop = train_size+dev_size+test_size
for i in tqdm(range(start,stop)):
  src_file_out.write(src_text[i])
  trgt_file_out.write(trgt_text[i])




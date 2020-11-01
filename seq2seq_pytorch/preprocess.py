import string
import pandas as pd
from sklearn.model_selection import train_test_split

#
#source_path = "../../TeamDatasets/az-en/az-en-tanzil/Tanzil.az-en.az"
#target_path = "../../TeamDatasets/az-en/az-en-tanzil/Tanzil.az-en.en"

#source_path = "../../TeamDatasets/az-tr/Tanzil.az-tr.az"
#target_path = "../../TeamDatasets/az-tr/Tanzil.az-tr.tr"

source_path = "../../TeamDatasets/en-tr/Tanzil.en-tr.tr"
target_path = "../../TeamDatasets/en-tr/Tanzil.en-tr.en"



source_txt = open(source_path, encoding='utf8').read().split('\n')

target_txt = open(target_path, encoding='utf8').read().split('\n')


target_txt = target_txt[:20000]
source_txt = source_txt[:20000]

def text_cleaning(data): 
    data = [s.translate(str.maketrans('','',string.punctuation)) for s in data]
    data = [' '.join(s.split()) for s in data]
    return data


target_txt = text_cleaning(target_txt)
source_txt = text_cleaning(source_txt)


raw_data = {'Source_lang': [sent for sent in source_txt],
            'Target_lang': [sent for sent in target_txt]}

df = pd.DataFrame(raw_data, columns=['Source_lang','Target_lang'])



train, test = train_test_split(df, test_size = 0.2)

train.to_csv('az_en_train20k.csv',index = False)
test.to_csv('az_en_test20k.csv',index = False)

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator,TabularDataset
import numpy as np
import spacy
import random
from utils import bleu, Translate
from tqdm import tqdm

checkpoint_path = "TestModels/testtest.pth.tar"
last_checkpoint = "TestModels/last_testtest.pth.tar"

set_source_to = "azerbaijani"
set_target_to = "english"

#sentence="Ad tayfasına da qardaşları Hudu peyğəmbər göndərdik O dedi “ Ey camaatım Allaha ibadət edin Sizin ondan başqa heç bir tanrınız yoxdur Məgər Allahın əzabından qorxmursunuz ”"
#sentence = "Məgər onlar Allahın xəlq etdiyi hər hansı bir şeyi müşahidə etməyiblərmi Onların kölgələri müti surətdə Allaha səcdə edərək sağasola əyilir "

#sentence = sentence.split()

#python -m spacy download en

spacy_eng = spacy.load("en")


tokenize_custom = lambda x: x.split()

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


src_tokenizer = None
trg_tokenizer = None

if set_source_to == "azerbaijani" or set_source_to == "turkish":
    src_tokenizer = tokenize_custom
elif set_source_to == "english":
    src_tokenizer = tokenize_eng

if set_target_to == "azerbaijani" or set_target_to == "turkish":
    trg_tokenizer = tokenize_custom
elif set_target_to == "english":
    trg_tokenizer = tokenize_eng

source_lang = Field(tokenize=src_tokenizer, lower=True, init_token="<sos>", eos_token="<eos>")

target_lang = Field(tokenize=trg_tokenizer, lower=True, init_token="<sos>", eos_token="<eos>")



fields = {'Source_lang': ('src',source_lang), 'Target_lang' : ('trg',target_lang)}

train_data, test_data = TabularDataset.splits(
        path='',
        train = 'az_en_train20k.csv',
        test = 'az_en_test20k.csv',
        format = 'csv',
        fields = fields
        )

source_lang.build_vocab(train_data, max_size=25000, min_freq=2)
target_lang.build_vocab(train_data, max_size=25000, min_freq=2)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):

        embedding = self.dropout(self.embedding(x))

        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell

class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Sequence2Sequence(nn.Module):
    def __init__(self, encoder, decoder):
        super(Sequence2Sequence, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target):
        batch_size = source.shape[1]
        #print(source.shape[1])
        target_len = target.shape[0]
        #print(source.shape[1])
        target_vocab_size = len(target_lang.vocab)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        hidden, cell = self.encoder(source)
        x = target[0]
        #print("x-->",x)
        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, hidden, cell)
            # Store next output prediction
            outputs[t] = output
            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)
            x = best_guess

        return outputs



# Training hyperparameters
num_epochs = 10
learning_rate = 0.001
batch_size = 32

# Model hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device set to: ",device)
#device="cpu"
input_size_encoder = len(source_lang.vocab)
input_size_decoder = len(target_lang.vocab)
output_size = len(target_lang.vocab)
encoder_embedding_size = 200
decoder_embedding_size = 200
# Hidden layer has to be same encoder and decoder
hidden_size = 512  
num_layers = 1
enc_dropout = 0
dec_dropout = 0


train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

network_enc = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

network_dec = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)

model = Sequence2Sequence(network_enc, network_dec).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = target_lang.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)




#Early Stopping variables below
training_loss = []
patience = 30
Max_BleuScore = None

counter = 0



#sentence_eng = "What shall I take other than Him gods whose intercession if the Merciful desires to afflict me cannot help me at all and they will never save me"

for epoch in tqdm(range(num_epochs), desc="Epoch Loop "):
    print(f"\n[Epoch {epoch} / {num_epochs}]")

    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)
        # Forward prop
        output = model(inp_data, target)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        # Back prop
        loss.backward()
        
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        training_loss.append(loss.item())
    
    avg_train_loss = np.average(training_loss)
    
    model.eval()
    BleuScore = bleu(test_data[:2000], model, source_lang, target_lang, device,max_length=50,generate_outputs=False)
    #print(f"Bleu score {score*100:.2f}")
    print(f'\n[epoch:{epoch}] -->>'+f'training loss is: {avg_train_loss:.5f}' + f' Bleu Score is: {BleuScore*100:.5f}')
    #reset losses for next epoch
    training_loss = []
    
    #Earlystop check
    if Max_BleuScore == None:
        Max_BleuScore = BleuScore
        #print("First min loss",min_loss)
        print("saving first model")
        torch.save(model,checkpoint_path)
    elif BleuScore > Max_BleuScore:
        Max_BleuScore = BleuScore
        print("New max bleu score found , saving model")
        torch.save(model,checkpoint_path)
        counter = 0
    else:
        counter +=1
        print("counter is:",counter)
        if counter == patience:
            print("early stopping as patience is reached")
            print("saving last model")
            torch.save(model,last_checkpoint)
            break
    
    #saving the last epoch model just in case
    if epoch+1 == num_epochs:
        print("saving last model")
        torch.save(model,last_checkpoint)
    
    #translated_sentence = Translate(model, sentence, source_lang, target_lang, device, max_length=50)

    #print(f"Translated example sentence: \n {translated_sentence}")

model = torch.load(checkpoint_path)
#model = torch.load(last_checkpoint)
#model.eval()

score = bleu(test_data[:2000], model, source_lang, target_lang, device,max_length=50,generate_outputs=True)
print(f"Bleu score {score*100:.3f}")

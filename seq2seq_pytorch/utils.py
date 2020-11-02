import torch
from torchtext.data.metrics import bleu_score


def Translate(model, sentence, source_lang, target_lang, device, max_length):
    
    # Add <SOS> and <EOS> in beginning and end respectively
    if(sentence[0] != "<sos>"):
        sentence.insert(0, source_lang.init_token)
        sentence.append(source_lang.eos_token)

    # Convert sentence to tokens
    text_to_indices = [source_lang.vocab.stoi[token] for token in sentence]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)
        
    outputs = [target_lang.vocab.stoi["<sos>"]]

    for i in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            max_output = output.argmax(1).item()

        outputs.append(max_output)
        # if model predicts end of sentence then break
        if max_output == target_lang.vocab.stoi["<eos>"]:
            break

    translated_sentence = [target_lang.vocab.itos[idx] for idx in outputs]

    return translated_sentence




def bleu(data, model, source_lang, target_lang, device,max_length,generate_outputs):
    targets = []
    outputs = []
    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]
        
        prediction = Translate(model, src, source_lang, target_lang, device,max_length)
        # remove <sos> <eos> token
        prediction = prediction[1:-1]  

        targets.append([trg])
        outputs.append(prediction)
        
    if generate_outputs:
        writer = [' '.join(s) for s in outputs]
        with open("Outputs/tr_en_translated.txt", 'w',encoding='utf-8') as op:
            for sent in writer:    
                op.write(sent + '\n')
    #return corpus_bleu(targets, outputs)
    return bleu_score(outputs, targets)

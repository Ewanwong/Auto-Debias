import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn as nn
import torch.nn.functional as F
import regex as re
from tqdm import tqdm

pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# Initialize the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

"""
# example input batches
sentences_pairs = [["The president is a black man.",
                    "The president is a black woman."],
                    ["He is a doctor.",
                     "She is a doctor."]]

# tokenizer sentences
inputs = tokenizer([sent+tokenizer.pad_token for sent_pair in sentences_pairs for sent in sent_pair], padding=True, return_tensors='pt')
# print(inputs)

# Get the input IDs and labels
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
labels = input_ids.clone()
labels.masked_fill_(attention_mask == 0, -100)

# Forward pass through the model
outputs = model(**inputs, labels=labels)
loss = outputs[0]


# Print the loss
print("Cross-entropy loss:", loss.item())

"""

def get_loss(model, tokenizer, sentences_pairs, female_words, male_words, neutral_words):
    
    # LM loss

    # tokenizer sentences
    inputs = tokenizer([sent+tokenizer.pad_token for sent_pair in sentences_pairs for sent in sent_pair], padding=True, return_tensors='pt')
    # print(inputs)

    # Get the input IDs and labels
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = input_ids.clone()
    labels.masked_fill_(attention_mask == 0, -100)

    # Forward pass through the model
    outputs = model(**inputs, labels=labels) #TODO: add prefix
    loss = outputs[0]

    return loss

    

def construct_prefix_pairs(sentences_pairs, female_words, male_words, neutral_words):
    # construct pairs
    train_pairs_gender = [[], []]  # The doctor is a young man/woman.
    train_pairs_neutral = [[], []] # He is a doctor vs. She is a nurse
    train_pairs_gender_prior = [[], []] # He/She 

    for sent_pair in sentences_pairs:
        sent1, sent2 = sent_pair
        sent1_tokens, sent2_tokens = [tok.strip().lower() for tok in re.findall(pat, sent1)], [tok.strip().lower() for tok in re.findall(pat, sent2)] # case sensitive?
        assert len(sent1_tokens) == len(sent2_tokens)

        prefix_length = len(sent1_tokens[0])
        prefix = sent1_tokens[0]
        intervals = [0] # record where to put whitespace
        for i in range(1, len(sent1_tokens)):
            if prefix_length+len(sent1_tokens[i])+1 > len(sent1_tokens):
                intervals.append(0)
                prefix += sent1_tokens[i]
                prefix_length += len(sent1_tokens[i])
            elif prefix + ' ' + sent1_tokens[i] == sent1_tokens[:prefix_length+len(sent1_tokens[i])+1]:
                intervals.append(1)
                prefix = prefix + ' ' + sent1_tokens[i]
                prefix_length += len(sent1_tokens[i]) + 1
            else:
                intervals.append(0)
                prefix += sent1_tokens[i]
                prefix_length += len(sent1_tokens[i])
        
        # TODO: gender pair only once
        gender_present = False
        neutral_present = False
        for i in range(len(sent1_tokens)):

            if not gender_present and not neutral_present and sent1_tokens[i].lower() in female_words or sent1_tokens[i].lower() in male_words:
                sent1_prefix, sent2_prefix = '', ''

                if i != 0:
                    for j in range(i):
                        sent1_prefix += ' '*intervals[j] + sent1_tokens[j]
                        sent2_prefix += ' '*intervals[j] + sent2_tokens[j]
                
                train_pairs_gender[0].append(sent1_prefix)
                train_pairs_gender[1].append(sent2_prefix)
                gender_present = True

            elif not gender_present and neutral_present and sent1_tokens[i].lower() in female_words or sent1_tokens[i].lower() in male_words:
                sent1_prefix, sent2_prefix = '', ''

                if i != 0:
                    for j in range(i):
                        sent1_prefix += ' '*intervals[j] + sent1_tokens[j]
                        sent2_prefix += ' '*intervals[j] + sent2_tokens[j]
                
                train_pairs_gender_prior[0].append(sent1_prefix)
                train_pairs_gender_prior[1].append(sent2_prefix)
                gender_present = True

            if gender_present and sent1_tokens[i].lower() in neutral_words:
                sent1_prefix, sent2_prefix = '', ''

                if i != 0:
                    for j in range(i):
                        sent1_prefix += ' '*intervals[j] + sent1_tokens[j]
                        sent2_prefix += ' '*intervals[j] + sent2_tokens[j]
                
                train_pairs_neutral[0].append(sent1_prefix)
                train_pairs_neutral[1].append(sent2_prefix)
                neutral_present = True

    return train_pairs_gender, train_pairs_neutral, train_pairs_gender_prior

def gender_loss(model, tokenizer, train_pairs, male_words, female_words, jsd_model):
    sents1_prefix, sents2_prefix = train_pairs
    fine_tuning_male_vocab = get_fine_tuning_vocab(tokenizer, male_words)
    fine_tuning_female_vocab = get_fine_tuning_vocab(tokenizer, female_words)

    sents1_tokenized = tokenizer(sents1_prefix, padding=True, return_tensors='pt')
    sents2_tokenized = tokenizer(sents2_prefix, padding=True, return_tensors='pt')

    sents1_predictions = model(**sents1_tokenized) #TODO: add prefix
    sents2_predictions = model(**sents2_tokenized)

    sents1_predictions_logits = sents1_predictions.prediction_logits[:, fine_tuning_male_vocab]
    sents2_predictions_logits = sents2_predictions.prediction_logits[:, fine_tuning_female_vocab]

    loss = jsd_model(sents1_predictions_logits, sents2_predictions_logits)

    return loss

def neutral_loss(model, tokenizer, train_pairs_neutral, neutral_words, jsd_model):
    sents1_prefix, sents2_prefix = train_pairs_neutral
    fine_tuning_vocab = get_fine_tuning_vocab(tokenizer, neutral_words)


    sents1_tokenized = tokenizer(sents1_prefix, padding=True, return_tensors='pt')
    sents2_tokenized = tokenizer(sents2_prefix, padding=True, return_tensors='pt')

    sents1_predictions = model(**sents1_tokenized) #TODO: add prefix
    sents2_predictions = model(**sents2_tokenized)

    sents1_predictions_logits = sents1_predictions.prediction_logits[:, fine_tuning_vocab]
    sents2_predictions_logits = sents2_predictions.prediction_logits[:, fine_tuning_vocab]

    loss = jsd_model(sents1_predictions_logits, sents2_predictions_logits)

    return loss

def get_fine_tuning_vocab(tokenizer, words_list):
    return tokenizer.convert_tokens_to_ids(words_list)

class JSD(nn.Module):
    def __init__(self,reduction='batchmean'):
        super(JSD, self).__init__()
        self.reduction = reduction

    def forward(self, net_1_logits, net_2_logits):
        net_1_probs = F.softmax(net_1_logits, dim=1)
        net_2_probs= F.softmax(net_2_logits, dim=1)

        total_m = 0.5 * (net_1_probs + net_2_probs)
        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), total_m, reduction=self.reduction) 
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), total_m, reduction=self.reduction) 
     
        return (0.5 * loss) 

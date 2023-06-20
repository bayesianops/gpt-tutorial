import random
import os
from cmdstanpy import CmdStanModel
import cmdstanpy
import json


def encoder_decoder_1_indexed(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:(i + 1) for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c]  for c in s]  # encoder: take a string, output list of ints
    itos = { (i + 1):ch for i,ch in enumerate(chars) }
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder; take a list of ints, output string
    return encode, decode


def get_data_batch(data, batch_size, block_size):
    idx = random.sample(range(0, len(data) - block_size), batch_size)
    x = [data[i:i+block_size] for i in idx]
    y = [data[i+1:i+block_size+1] for i in idx]
    return x, y


## Read Shakespeare data
with open('../data/tinyshakespeare/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

## Set up encoder and decoder for this text
encode, decode = encoder_decoder_1_indexed(text)

## Split data into training and validation sets
data_full = encode(text)
n = int(0.9 * len(data_full))   # first 90% will be train, rest validation
data_train = data_full[:n]
data_val = data_full[n:]


model_07 = CmdStanModel(stan_file=os.path.join('..', 'stan', '07-skip-connections.stan'))

with open('../cache/model_07_data.json', 'r') as f:
    data = json.load(f)

optimum_07 = cmdstanpy.from_csv('../cache/07-skip-connections.csv')

print("Cached tokens")
print("************************************************************")
print(decode(optimum_07.stan_variable('new_tokens')))
print("************************************************************")


gq_07 = model_07.generate_quantities(data=data, previous_fit=optimum_07)

print("Newly generated tokens")
print("************************************************************")
print(decode(gq_07.stan_variable('new_tokens')[0]))
print("************************************************************")



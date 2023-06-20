import random
import os
from cmdstanpy import CmdStanModel
import cmdstanpy

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

vocab_size = len(set(text))   # total number of characters in the text
batch_size = 32  # how many independent sequences will we process in parallel;  B
block_size = 8   # what is the maximum context length for predictions?;         T
n_embed = 32     # embedding size
n_head = 2       # number of heads => head_size = n_embed / head_size

xb, yb = get_data_batch(data_train, batch_size, block_size)
xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
data = {
    'vocab_size': vocab_size,
    'batch_size': batch_size,
    'block_size': block_size,
    'n_embed': n_embed,
    'n_head': n_head,
    'xb': xb,
    'yb': yb,
    'xb_val': xb_val,
    'yb_val': yb_val,
    'max_new_tokens': 500
}

optimum_07 = model_07.optimize(data=data, show_console=True, iter=1, init_alpha=0.0001, algorithm="LBFGS", inits=0.1)
for step in range(10000):
    print("step = ", step)
    xb, yb = get_data_batch(data_train, batch_size, block_size)
    xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
    data['xb'] = xb
    data['yb'] = yb
    data['xb_val'] = xb_val
    data['yb_val'] = yb_val
    optimum_07 = model_07.optimize(data = data, show_console=(step % 100 == 0),
                                   iter=1, init_alpha=0.0001, algorithm="LBFGS",
                                   inits=optimum_07.stan_variables())

print(optimum_07.stan_variable('loss'))
print(optimum_07.stan_variable('loss_validation'))

print(decode(optimum_07.stan_variable('new_tokens')))
    

cmdstanpy.write_stan_json('../cache/model_07_data.json', data)
optimum_07.save_csvfiles('../cache/')

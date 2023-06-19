import random
import os
from cmdstanpy import CmdStanModel

verbose = True

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


def print_shakespeare_data_info(text):
    print("************************************************************")
    print("Some information about the Shakespeare data")
    print("* length of dataset in characters: ", len(text))
    chars = sorted(list(set(text)))
    print("* number of unique characters: ", len(chars))
    print("* characeters: ", ''.join(chars))
    print("* Initial 1000 characeters of dataset")
    print("------------------------------------------------------------")
    print(text[:1000])
    print("------------------------------------------------------------")
    print()


def print_encode_decode_info(encode, decode, text):
    print("************************************************************")
    print("Some information using encode and decode")
    print("encoding 'abc': ", encode('abc'))
    print("decoding encode('abc'): ", decode(encode('abc')))
    print("------------------------------------------------------------")
    print("initial 10 characeters:           ", text[:10])
    print("encoding the first 10 characters: ", encode(text[:10]))
    print("                       expecting: ", [19, 48, 57, 58, 59, 2, 16, 48, 59, 48])
    print("------------------------------------------------------------")
    print()


def print_batch_data_info(xb, yb, xb_val, yb_val):
    print("************************************************************")
    print("Some information about the training and validation data")
    print("input xb.len:          ", len(xb), ", ", len(xb[0]))
    print("target yb.len:         ", len(yb), ", ", len(yb[0]))
    print("decode(xb[0]):         ", decode(xb[0]))
    print("decode(yb[0]):         ", decode(yb[0]))
    print("input xb_val.len:      ", len(xb_val), ", ", len(xb_val[0]))
    print("target yb_val.len:     ", len(yb_val), ", ", len(yb_val[0]))    
    print("decode(xb_val[0]):     ", decode(xb_val[0]))
    print("decode(yb_val[0]):     ", decode(yb_val[0]))
    print()


################################################################################
## Start of script
################################################################################


## Read Shakespeare data
with open('../data/tinyshakespeare/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

if (verbose):
    print_shakespeare_data_info(text)


## Set up encoder and decoder for this text
encode, decode = encoder_decoder_1_indexed(text)

if (verbose):
    print_encode_decode_info(encode, decode, text)


## Split data into training and validation sets
data_full = encode(text)
n = int(0.9 * len(data_full))   # first 90% will be train, rest validation
data_train = data_full[:n]
data_val = data_full[n:]


xb, yb = get_data_batch(data_train, 10, 8)
xb_val, yb_val = get_data_batch(data_val, 10, 8)

if (verbose):
    print_batch_data_info(xb, yb, xb_val, yb_val)


## Start of Language models
vocab_size = len(set(text))   # total number of characters in the text
batch_size = 32  # how many independent sequences will we process in parallel;  B
block_size = 8   # what is the maximum context length for predictions?;         T


############################################################
## 01: bigram model
model_01 = CmdStanModel(stan_file=os.path.join('..', 'stan', '01-bigram.stan'))


xb, yb = get_data_batch(data_train, batch_size, block_size)
xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
data = {
    'vocab_size': vocab_size,
    'batch_size': batch_size,
    'block_size': block_size,
    'xb': xb,
    'yb': yb,
    'xb_val': xb_val,
    'yb_val': yb_val,
    'max_new_tokens': 500
}


## This is wrong. Why? 
optimum_01 = model_01.optimize(data=data, show_console=True, algorithm="LBFGS")
print(decode(optimum_01.stan_variable('new_tokens')))

gq_01 = model_01.generate_quantities(data=data, previous_fit=optimum_01)
print(decode(gq_01.stan_variable('new_tokens')[0]))


## Stochastic LBFGS
optimum_01 = model_01.optimize(data=data, show_console=True, iter=1, init_alpha=0.0001, algorithm="LBFGS")

for step in range(1000):
    print("step = ", step)
    xb, yb = get_data_batch(data_train, batch_size, block_size)
    xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
    data['xb'] = xb
    data['yb'] = yb
    data['xb_val'] = xb_val
    data['yb_val'] = yb_val
    optimum_01 = model_01.optimize(data = data, show_console=(step % 100 == 0),
                                   iter=1, init_alpha=0.0001, algorithm="LBFGS",
                                   inits=optimum_01.stan_variables())

print(optimum_01.stan_variable('loss'))
print(optimum_01.stan_variable('loss_validation'))   
    
print(decode(optimum_01.stan_variable('new_tokens')))

gq_01 = model_01.generate_quantities(data=data, previous_fit=optimum_01)
print(decode(gq_01.stan_variable('new_tokens')[0]))

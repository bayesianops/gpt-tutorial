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


############################################################
## 01: bigram model
##     Only look at the last character to predict the next character.
##     Look at how to compute loss
##     Look at optimization
model_01 = CmdStanModel(stan_file=os.path.join('..', 'stan', '01-bigram.stan'))


vocab_size = len(set(text))   # total number of characters in the text
batch_size = 32  # how many independent sequences will we process in parallel;  B
block_size = 8   # what is the maximum context length for predictions?;         T

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
## Look at loss and loss_val
optimum_01 = model_01.optimize(data=data, show_console=True, algorithm="LBFGS")
print(optimum_01.stan_variable('loss'))
print(optimum_01.stan_variable('loss_validation'))

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



############################################################
## 02: embedding
##     Instead of directly using logits of vocab_size, estimate a vector of size n_embed (> vocab_size)
##     To get it back to vocab_size, matrix multiply
model_02 = CmdStanModel(stan_file=os.path.join('..', 'stan', '02-different-embedding-size.stan'))

vocab_size = len(set(text))   # total number of characters in the text
batch_size = 32  # how many independent sequences will we process in parallel;  B
block_size = 8   # what is the maximum context length for predictions?;         T
n_embed = 32     # embedding size

xb, yb = get_data_batch(data_train, batch_size, block_size)
xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
data = {
    'vocab_size': vocab_size,
    'batch_size': batch_size,
    'block_size': block_size,
    'n_embed': n_embed,
    'xb': xb,
    'yb': yb,
    'xb_val': xb_val,
    'yb_val': yb_val,
    'max_new_tokens': 500
}

optimum_02 = model_02.optimize(data=data, show_console=True, iter=1, init_alpha=0.0001, algorithm="LBFGS")

for step in range(1000):
    print("step = ", step)
    xb, yb = get_data_batch(data_train, batch_size, block_size)
    xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
    data['xb'] = xb
    data['yb'] = yb
    data['xb_val'] = xb_val
    data['yb_val'] = yb_val
    optimum_02 = model_02.optimize(data = data, show_console=(step % 100 == 0),
                                   iter=1, init_alpha=0.0001, algorithm="LBFGS",
                                   inits=optimum_02.stan_variables())

print(optimum_02.stan_variable('loss'))
print(optimum_02.stan_variable('loss_validation'))

print(decode(optimum_02.stan_variable('new_tokens')))

gq_02 = model_02.generate_quantities(data=data, previous_fit=optimum_02)
print(decode(gq_02.stan_variable('new_tokens')[0]))


############################################################
## 03: position encoding
##     Use positional encoding. Acts as an "intercept" on the logit scale for each position in block_size.
##     Only uses the last character; take a look at the generation code
model_03 = CmdStanModel(stan_file=os.path.join('..', 'stan', '03-positional-encoding.stan'))

vocab_size = len(set(text))   # total number of characters in the text
batch_size = 32  # how many independent sequences will we process in parallel;  B
block_size = 8   # what is the maximum context length for predictions?;         T
n_embed = 32     # embedding size

xb, yb = get_data_batch(data_train, batch_size, block_size)
xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
data = {
    'vocab_size': vocab_size,
    'batch_size': batch_size,
    'block_size': block_size,
    'n_embed': n_embed,
    'xb': xb,
    'yb': yb,
    'xb_val': xb_val,
    'yb_val': yb_val,
    'max_new_tokens': 500
}

optimum_03 = model_03.optimize(data=data, show_console=True, iter=1, init_alpha=0.0001, algorithm="LBFGS", inits=0.1)

for step in range(1000):
    print("step = ", step)
    xb, yb = get_data_batch(data_train, batch_size, block_size)
    xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
    data['xb'] = xb
    data['yb'] = yb
    data['xb_val'] = xb_val
    data['yb_val'] = yb_val
    optimum_03 = model_03.optimize(data = data, show_console=(step % 100 == 0),
                                   iter=1, init_alpha=0.0001, algorithm="LBFGS",
                                   inits=optimum_03.stan_variables())

print(optimum_03.stan_variable('loss'))
print(optimum_03.stan_variable('loss_validation'))

print(decode(optimum_03.stan_variable('new_tokens')))

gq_03 = model_03.generate_quantities(data=data, previous_fit=optimum_03)
print(decode(gq_03.stan_variable('new_tokens')[0]))


############################################################
## 04: self-attention

model_04 = CmdStanModel(stan_file=os.path.join('..', 'stan', '04-self-attention.stan'))

vocab_size = len(set(text))   # total number of characters in the text
batch_size = 32  # how many independent sequences will we process in parallel;  B
block_size = 8   # what is the maximum context length for predictions?;         T
n_embed = 32     # embedding size
# For self attention, head_size = embedding size

xb, yb = get_data_batch(data_train, batch_size, block_size)
xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
data = {
    'vocab_size': vocab_size,
    'batch_size': batch_size,
    'block_size': block_size,
    'n_embed': n_embed,
    'xb': xb,
    'yb': yb,
    'xb_val': xb_val,
    'yb_val': yb_val,
    'max_new_tokens': 500
}

optimum_04 = model_04.optimize(data=data, show_console=True, iter=1, init_alpha=0.0001, algorithm="LBFGS", inits=0.1)
for step in range(1000):
    print("step = ", step)
    xb, yb = get_data_batch(data_train, batch_size, block_size)
    xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
    data['xb'] = xb
    data['yb'] = yb
    data['xb_val'] = xb_val
    data['yb_val'] = yb_val
    optimum_04 = model_04.optimize(data = data, show_console=(step % 100 == 0),
                                   iter=1, init_alpha=0.0001, algorithm="LBFGS",
                                   inits=optimum_04.stan_variables())

print(optimum_04.stan_variable('loss'))
print(optimum_04.stan_variable('loss_validation'))

print(decode(optimum_04.stan_variable('new_tokens')))

gq_04 = model_04.generate_quantities(data=data, previous_fit=optimum_04)
print(decode(gq_04.stan_variable('new_tokens')[0]))



############################################################
## 05: multi-head self-attention
model_05 = CmdStanModel(stan_file=os.path.join('..', 'stan', '05-multi-headed-self-attention.stan'))

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

optimum_05 = model_05.optimize(data=data, show_console=True, iter=1, init_alpha=0.0001, algorithm="LBFGS", inits=0.1)
for step in range(1000):
    print("step = ", step)
    xb, yb = get_data_batch(data_train, batch_size, block_size)
    xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
    data['xb'] = xb
    data['yb'] = yb
    data['xb_val'] = xb_val
    data['yb_val'] = yb_val
    optimum_05 = model_05.optimize(data = data, show_console=(step % 100 == 0),
                                   iter=1, init_alpha=0.0001, algorithm="LBFGS",
                                   inits=optimum_05.stan_variables())

print(optimum_05.stan_variable('loss'))
print(optimum_05.stan_variable('loss_validation'))

print(decode(optimum_05.stan_variable('new_tokens')))

gq_05 = model_05.generate_quantities(data=data, previous_fit=optimum_05)
print(decode(gq_05.stan_variable('new_tokens')[0]))



############################################################
## 06: feed forward
model_06 = CmdStanModel(stan_file=os.path.join('..', 'stan', '06-feed-forward.stan'))

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

optimum_06 = model_06.optimize(data=data, show_console=True, iter=1, init_alpha=0.0001, algorithm="LBFGS", inits=0.1)
for step in range(1000):
    print("step = ", step)
    xb, yb = get_data_batch(data_train, batch_size, block_size)
    xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
    data['xb'] = xb
    data['yb'] = yb
    data['xb_val'] = xb_val
    data['yb_val'] = yb_val
    optimum_06 = model_06.optimize(data = data, show_console=(step % 100 == 0),
                                   iter=1, init_alpha=0.0001, algorithm="LBFGS",
                                   inits=optimum_06.stan_variables())

print(optimum_06.stan_variable('loss'))
print(optimum_06.stan_variable('loss_validation'))

print(decode(optimum_06.stan_variable('new_tokens')))

gq_06 = model_06.generate_quantities(data=data, previous_fit=optimum_06)
print(decode(gq_06.stan_variable('new_tokens')[0]))



############################################################
## 07: skip connections
##     Math trick to have gradients work through
## PROMISING AS A MID POINT
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
for step in range(1000):
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

gq_07 = model_07.generate_quantities(data=data, previous_fit=optimum_07)
print(decode(gq_07.stan_variable('new_tokens')[0]))


############################################################
## 08: larger feed forward layer
model_08 = CmdStanModel(stan_file=os.path.join('..', 'stan', '08-larger-feed-forward-layer.stan'))

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
    'xb_val': xb,
    'yb_val': yb,
    'max_new_tokens': 500
}

optimum_08 = model_08.optimize(data=data, show_console=True, iter=1, init_alpha=0.0001, algorithm="LBFGS", inits=0.1)
for step in range(1000):
    print("step = ", step)
    xb, yb = get_data_batch(data_train, batch_size, block_size)
    xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
    data['xb'] = xb
    data['yb'] = yb
    data['xb_val'] = xb_val
    data['yb_val'] = yb_val
    optimum_08 = model_08.optimize(data = data, show_console=(step % 100 == 0),
                                   iter=1, init_alpha=0.0001, algorithm="LBFGS",
                                   inits=optimum_08.stan_variables())

print(optimum_08.stan_variable('loss'))
print(optimum_08.stan_variable('loss_validation'))

print(decode(optimum_08.stan_variable('new_tokens')))

gq_08 = model_08.generate_quantities(data=data, previous_fit=optimum_08)
print(decode(gq_08.stan_variable('new_tokens')[0]))



############################################################
## 09: layer norm
model_09 = CmdStanModel(stan_file=os.path.join('..', 'stan', '09-layer-norm.stan'))

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

optimum_09 = model_09.optimize(data=data, show_console=True, iter=1, init_alpha=0.0001, algorithm="LBFGS", inits=0.1)
for step in range(1000):
    print("step = ", step)
    xb, yb = get_data_batch(data_train, batch_size, block_size)
    xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
    data['xb'] = xb
    data['yb'] = yb
    data['xb_val'] = xb_val
    data['yb_val'] = yb_val
    optimum_09 = model_09.optimize(data = data, show_console=(step % 100 == 0),
                                   iter=1, init_alpha=0.0001, algorithm="LBFGS",
                                   inits=optimum_09.stan_variables())

print(optimum_09.stan_variable('loss'))
print(optimum_09.stan_variable('loss_validation'))

print(decode(optimum_09.stan_variable('new_tokens')))

gq_09 = model_09.generate_quantities(data=data, previous_fit=optimum_09)
print(decode(gq_09.stan_variable('new_tokens')[0]))


############################################################
## 10: blocks
model_10 = CmdStanModel(stan_file=os.path.join('..', 'stan', '10-blocks.stan'))

vocab_size = len(set(text))   # total number of characters in the text
batch_size = 32  # how many independent sequences will we process in parallel;  B
block_size = 8   # what is the maximum context length for predictions?;         T
n_embed = 32     # embedding size
n_head = 2       # number of heads => head_size = n_embed / head_size
n_layer = 2      # number of transformer blocks


xb, yb = get_data_batch(data_train, batch_size, block_size)
xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
data = {
    'vocab_size': vocab_size,
    'batch_size': batch_size,
    'block_size': block_size,
    'n_embed': n_embed,
    'n_head': n_head,
    'n_layer': n_layer,
    'xb': xb,
    'yb': yb,
    'xb_val': xb_val,
    'yb_val': yb_val,
    'max_new_tokens': 500
}

optimum_10 = model_10.optimize(data=data, show_console=True, iter=1, init_alpha=0.0001, algorithm="LBFGS", inits=0.1)
for step in range(1000):
    print("step = ", step)
    xb, yb = get_data_batch(data_train, batch_size, block_size)
    xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
    data['xb'] = xb
    data['yb'] = yb
    data['xb_val'] = xb_val
    data['yb_val'] = yb_val
    optimum_10 = model_10.optimize(data = data, show_console=(step % 100 == 0),
                                   iter=1, init_alpha=0.0001, algorithm="LBFGS",
                                   inits=optimum_10.stan_variables())

print(optimum_10.stan_variable('loss'))
print(optimum_10.stan_variable('loss_validation'))

print(decode(optimum_10.stan_variable('new_tokens')))

gq_10 = model_10.generate_quantities(data=data, previous_fit=optimum_10)
print(decode(gq_10.stan_variable('new_tokens')[0]))


############################################################
## 11: dropout
model_11 = CmdStanModel(stan_file=os.path.join('..', 'stan', '11-dropout.stan'))

vocab_size = len(set(text))   # total number of characters in the text
batch_size = 32  # how many independent sequences will we process in parallel;  B
block_size = 8   # what is the maximum context length for predictions?;         T
n_embed = 32     # embedding size
n_head = 2       # number of heads => head_size = n_embed / head_size
n_layer = 2      # number of transformer blocks
dropout = 0.2    # proportion to drop out


xb, yb = get_data_batch(data_train, batch_size, block_size)
xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
data = {
    'vocab_size': vocab_size,
    'batch_size': batch_size,
    'block_size': block_size,
    'n_embed': n_embed,
    'n_head': n_head,
    'n_layer': n_layer,
    'dropout': dropout,
    'xb': xb,
    'yb': yb,
    'xb_val': xb_val,
    'yb_val': yb_val,
    'max_new_tokens': 500
}

optimum_11 = model_11.optimize(data=data, show_console=True, iter=1, init_alpha=0.0001, algorithm="LBFGS", inits=0.1)
for step in range(1000):
    print("step = ", step)
    xb, yb = get_data_batch(data_train, batch_size, block_size)
    xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
    data['xb'] = xb
    data['yb'] = yb
    data['xb_val'] = xb_val
    data['yb_val'] = yb_val
    optimum_11 = model_11.optimize(data = data, show_console=(step % 100 == 0),
                                   iter=1, init_alpha=0.0001, algorithm="LBFGS",
                                   inits=optimum_11.stan_variables())

print(optimum_11.stan_variable('loss'))
print(optimum_11.stan_variable('loss_validation'))

print(decode(optimum_11.stan_variable('new_tokens')))

gq_11 = model_11.generate_quantities(data=data, previous_fit=optimum_11)
print(decode(gq_11.stan_variable('new_tokens')[0]))


############################################################
## 12: final - projection
model_12 = CmdStanModel(stan_file=os.path.join('..', 'stan', '12-final.stan'))

vocab_size = len(set(text))   # total number of characters in the text
batch_size = 32  # how many independent sequences will we process in parallel;  B
block_size = 8   # what is the maximum context length for predictions?;         T
n_embed = 32     # embedding size
n_head = 2       # number of heads => head_size = n_embed / head_size
n_layer = 2      # number of transformer blocks
dropout = 0.2    # proportion to drop out


xb, yb = get_data_batch(data_train, batch_size, block_size)
xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
data = {
    'vocab_size': vocab_size,
    'batch_size': batch_size,
    'block_size': block_size,
    'n_embed': n_embed,
    'n_head': n_head,
    'n_layer': n_layer,
    'dropout': dropout,
    'xb': xb,
    'yb': yb,
    'xb_val': xb_val,
    'yb_val': yb_val,
    'max_new_tokens': 500
}

optimum_12 = model_12.optimize(data=data, show_console=True, iter=1, init_alpha=0.0001, algorithm="LBFGS", inits=0.1)
for step in range(1000):
    print("step = ", step)
    xb, yb = get_data_batch(data_train, batch_size, block_size)
    xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
    data['xb'] = xb
    data['yb'] = yb
    data['xb_val'] = xb_val
    data['yb_val'] = yb_val
    optimum_12 = model_12.optimize(data = data, show_console=(step % 100 == 0),
                                   iter=1, init_alpha=0.0001, algorithm="LBFGS",
                                   inits=optimum_12.stan_variables())

print(optimum_12.stan_variable('loss'))
print(optimum_12.stan_variable('loss_validation'))

print(decode(optimum_12.stan_variable('new_tokens')))

gq_12 = model_12.generate_quantities(data=data, previous_fit=optimum_12)
print(decode(gq_12.stan_variable('new_tokens')[0]))




## Try something bigger, staying under 16 GB ram on my laptop
vocab_size = len(set(text))
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
n_embed = 128
n_head = 4
n_layer = 4
dropout = 0.2

data = {
    'vocab_size': vocab_size,
    'batch_size': batch_size,
    'block_size': block_size,
    'n_embed': n_embed,
    'n_head': n_head,
    'n_layer': n_layer,
    'dropout': dropout,
    'xb': xb,
    'yb': yb,
    'xb_val': xb_val,
    'yb_val': yb_val,
    'max_new_tokens': 500
}

optimum_12 = model_12.optimize(data=data, show_console=True, iter=1, init_alpha=0.0001, algorithm="LBFGS", inits=0.1)
for step in range(1000):
    print("step = ", step)
    xb, yb = get_data_batch(data_train, batch_size, block_size)
    xb_val, yb_val = get_data_batch(data_val, batch_size, block_size)
    data['xb'] = xb
    data['yb'] = yb
    data['xb_val'] = xb_val
    data['yb_val'] = yb_val
    optimum_12 = model_12.optimize(data = data, show_console=(step % 100 == 0),
                                   iter=1, init_alpha=0.0001, algorithm="LBFGS",
                                   inits=optimum_12.stan_variables())

print(optimum_12.stan_variable('loss'))
print(optimum_12.stan_variable('loss_validation'))

print(decode(optimum_12.stan_variable('new_tokens')))

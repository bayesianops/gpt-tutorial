library(cmdstanr)
library(tidyverse)



encoder_decoder_1_indexed <- function(text) {
    ## this matches the python implementation. 
    chars = c('\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')
    ##sort(unique(text))
    vocab_size = length(chars)
    chars_factor = factor(chars)
    
    encode <- function(s) {
        return(as.numeric(factor(s, levels = chars)))
    }
    decode <- function(i) {
        return(as.character(factor(i, levels = 1:65, labels = chars)))
    }
    return(list(encode = encode, decode = decode))
}

get_data_batch <- function(data, batch_size, block_size) {
    idx = sample(1:(length(data) - block_size), batch_size, replace=TRUE)
    x = lapply(idx, function(i) {data[i:(i+block_size)]})
    y = lapply(idx, function(i) {data[(i+1):(i+block_size+1)]})
    return(list(x = x, y = y))
}


filename = '../data/tinyshakespeare/input.txt'
text = str_split_1(readChar(filename, file.info(filename)$size), "")


encode = encoder_decoder_1_indexed(text)$encode
decode = encoder_decoder_1_indexed(text)$decode


## Split data into training and validation sets
data_full = encode(text)
n = floor(0.9 * length(data_full))   # first 90% will be train, rest validation
data_train = data_full[1:n]
data_val = data_full[(n + 1): length(data_full)]


train = get_data_batch(data_train, 10, 8)
xb = train$x
yb = train$y

val = get_data_batch(data_val, 10, 8)
xb_val = val$x
yb_val = val$y

############################################################
## 01: bigram model
##     Only look at the last character to predict the next character.
##     Look at how to compute loss
##     Look at optimization

model_01 = cmdstan_model("../stan/01-bigram.stan")

data = list(
    
)

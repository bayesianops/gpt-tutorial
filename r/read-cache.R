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



filename = '../data/tinyshakespeare/input.txt'
text = str_split_1(readChar(filename, file.info(filename)$size), "")


encode = encoder_decoder_1_indexed(text)$encode
decode = encoder_decoder_1_indexed(text)$decode


## Split data into training and validation sets
data_full = encode(text)
n = floor(0.9 * length(data_full))   # first 90% will be train, rest validation
data_train = data_full[1:n]
data_val = data_full[(n + 1): length(data_full)]


model_07 = cmdstan_model("../stan/07-skip-connections.stan")

data = jsonlite::fromJSON('../cache/model_07_data.json')

optimum_07 = cmdstanr::as_cmdstan_fit('../cache/07-skip-connections.csv')

cat(paste(decode(as.numeric(optimum_07$draws('new_tokens'))), collapse=""))



gq_07 = model_07$generate_quantities(data = data, fitted_params = optimum_07$draws())

print("Newly generated tokens")
print("************************************************************")
cat(paste(decode(as.numeric(gq_07$draws('new_tokens'))), collapse=""))
print("************************************************************")




data {
  int<lower = 1> vocab_size;
  int<lower = 1> batch_size;
  int<lower = 1> block_size;
  array[batch_size, block_size] int<lower = 1, upper = vocab_size> xb;
  array[batch_size, block_size] int<lower = 1, upper = vocab_size> yb;

  array[batch_size, block_size] int<lower = 1, upper = vocab_size> xb_val;
  array[batch_size, block_size] int<lower = 1, upper = vocab_size> yb_val;

  int<lower = 0> max_new_tokens;
}
parameters {
  array[vocab_size] vector[vocab_size] token_embedding;
}
transformed parameters {
  real loss = 0;
  for (b in 1:batch_size) {
    for (t in 1:block_size) {
      loss += categorical_logit_lpmf(yb[b, t] | token_embedding[xb[b, t]]);
    }
  }
  loss /= batch_size * block_size;
}
model {
  target += loss;
}
generated quantities {
  real loss_validation = 0;
  for (b in 1:batch_size) {
    for (t in 1:block_size) {
      loss_validation += categorical_logit_lpmf(yb_val[b, t] | token_embedding[xb_val[b, t]]);
    }
  }
  loss_validation /= batch_size * block_size;
  print("************************************************************");
  print("train loss ", -loss, ", val loss ", -loss_validation);
  print("************************************************************");
  
  array[max_new_tokens] int<lower = 1, upper = vocab_size> new_tokens;
  new_tokens[1] = 1;
  for (n in 2:max_new_tokens) {
    new_tokens[n] = categorical_logit_rng(token_embedding[new_tokens[n - 1]]);
  }
}

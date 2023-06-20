functions {
  vector lm_head(vector x, matrix A, vector b) {
    return A * x + b;
  }
}
data {
  int<lower = 1> vocab_size;           // e.g. 65
  int<lower = 1> batch_size;           // e.g. 32
  int<lower = 1> block_size;           // e.g. 8
  int<lower = 1> n_embed;              // e.g. 32

  array[batch_size, block_size] int<lower = 1, upper = vocab_size> xb;
  array[batch_size, block_size] int<lower = 1, upper = vocab_size> yb;

  array[batch_size, block_size] int<lower = 1, upper = vocab_size> xb_val;
  array[batch_size, block_size] int<lower = 1, upper = vocab_size> yb_val;
  
  int<lower = 0> max_new_tokens;
}
parameters {
  array[vocab_size] vector[n_embed] token_embedding;        // 02 - change in token_embedding size
  matrix[vocab_size, n_embed] lm_head_multiplier;           // 02 - new parameters
  vector[vocab_size] lm_head_offset;                        // 02 - new parameters

  array[block_size] vector[n_embed] position_embedding;     // 03 - new parameter
}
transformed parameters {
  array[batch_size, block_size] vector[n_embed] x;
  for (b in 1:batch_size) {
    for (t in 1:block_size) {
      x[b, t] = token_embedding[xb[b, t]] + position_embedding[t];
    }
  }
  
  real loss = 0;  
  for (b in 1:batch_size) {
    for (t in 1:block_size) {
      vector[vocab_size] logits = lm_head(x[b, t], lm_head_multiplier, lm_head_offset);
      
      loss += categorical_logit_lpmf(yb[b, t] | logits);
    }
  }
  loss /= batch_size * block_size;
}
model {
  target += loss;
}
generated quantities {
  array[batch_size, block_size] vector[n_embed] x_val;
  for (b in 1:batch_size) {
    for (t in 1:block_size) {
      x_val[b, t] = token_embedding[xb_val[b, t]] + position_embedding[t];
    }
  }
  
  real loss_validation = 0;
  for (b in 1:batch_size) {
    for (t in 1:block_size) {
      vector[vocab_size] logits = lm_head(x_val[b, t], lm_head_multiplier, lm_head_offset);

      loss_validation += categorical_logit_lpmf(yb_val[b, t] | logits);
    }
  }
  loss_validation /= batch_size * block_size;
  print("************************************************************");
  print("train loss ", -loss, ", val loss ", -loss_validation);
  print("************************************************************");

  
  array[max_new_tokens] int<lower = 1, upper = vocab_size> new_tokens;
  new_tokens[1] = 1;
  for (n in 2:max_new_tokens) {
    vector[n_embed] x_new = token_embedding[new_tokens[n - 1]] + position_embedding[min(n - 1, block_size)];
    vector[vocab_size] logits = lm_head(x_new, lm_head_multiplier, lm_head_offset);
    new_tokens[n] = categorical_logit_rng(logits);
  }
}

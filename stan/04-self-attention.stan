functions {
  vector lm_head(vector x, matrix A, vector b) {
    return A * x + b;
  }

  // self_attention for a single block
  array[] vector self_attention(array[] vector x,  // block_size, vector[n_embed]
				matrix key,        // n_embed x head_size
				matrix query,      // n_embed x head_size
				matrix value) {    // n_embed x head_size
    // -> block_size, head_size
    int block_size = dims(x)[1];
    int n_embed = rows(key);
    int head_size = cols(key);


    matrix[block_size, n_embed] x_matrix;
    for (t in 1:block_size) {
      x_matrix[t, :] = x[t]';
    }
    matrix[block_size, head_size] k = x_matrix * key;
    matrix[block_size, head_size] q = x_matrix * query;
    matrix[block_size, head_size] v = x_matrix * value;

    // decoder block: wei is lower triangular matrix
    //   allows communication from different parts of the visible context
    //   e.g. 1st output only allows 1st x, 2nd output allows 1st and second, etc.
    matrix[block_size, block_size] wei = rep_matrix(0, block_size, block_size);
    matrix[block_size, block_size] tmp_wei = q * k' / sqrt(head_size);
    for (t in 1:block_size) {
      wei[t, 1:t] = softmax(tmp_wei[t, 1:t]')'; 
    }

    matrix[block_size, head_size] weighted_value = wei * v;
    array[block_size] vector[head_size] out;

    for (t in 1:block_size) {
      out[t] = weighted_value[t, :]';
    }
    return out;
  }

  // self attention for multiple blocks
  array[,] vector self_attention(array[,] vector x,        // batch_size, block_size, n_embed
				 matrix key,               // n_embed x head_size
				 matrix query,             // n_embed x head_size
				 matrix value) {           // n_embed x head_size
    // -> n_embed, block_size, head_size
    int batch_size = dims(x)[1];
    int block_size = dims(x)[2];
    int n_embed = rows(key);
    int head_size = cols(key);


    // loop over batch. Self attention works on blocks
    array[batch_size, block_size] vector[head_size] out;
    for (b in 1: batch_size) {
      out[b] = self_attention(x[b], key, query, value);
    }
    return out;
  }
}
data {
  int<lower = 1> vocab_size;           // e.g. 65
  int<lower = 1> batch_size;           // e.g. 32
  int<lower = 1> block_size;           // e.g. 8
  int<lower = 1> n_embed;              // e.g. 32
  int<lower = 1> head_size;            // e.g. n_embed = 32

  array[batch_size, block_size] int<lower = 1, upper = vocab_size> xb;
  array[batch_size, block_size] int<lower = 1, upper = vocab_size> yb;

  array[batch_size, block_size] int<lower = 1, upper = vocab_size> xb_val;
  array[batch_size, block_size] int<lower = 1, upper = vocab_size> yb_val;

  int<lower = 0> max_new_tokens;
}
transformed data {
}
parameters {
  array[vocab_size] vector[n_embed] token_embedding;        // 02 - change in token_embedding size
  matrix[vocab_size, n_embed] lm_head_multiplier;           // 02 - new parameters
  vector[vocab_size] lm_head_offset;                        // 02 - new parameters

  array[block_size] vector[n_embed] position_embedding;     // 03 - new parameter

  // Self Attention
  matrix[n_embed, head_size] key;                           // 04 - new parameter
  matrix[n_embed, head_size] query;                         // 04 - new parameter
  matrix[n_embed, head_size] value;                         // 04 - new parameter
}
transformed parameters {
  array[batch_size, block_size] vector[n_embed] x;
  for (b in 1:batch_size) {
    for (t in 1:block_size) {
      x[b, t] = token_embedding[xb[b, t]] + position_embedding[t];
    }
  }

  array[batch_size, block_size] vector[head_size] x_self_attention;
  x_self_attention = self_attention(x, key, query, value);

  real loss = 0;
  for (b in 1:batch_size) {
    for (t in 1:block_size) {
      vector[vocab_size] logits = lm_head(x_self_attention[b, t], lm_head_multiplier, lm_head_offset);
      loss += categorical_logit_lpmf(yb[b, t] | logits);
    }
  }
  loss /= batch_size * block_size;
}
model {
  target += loss;
}
generated quantities {
  real loss_validation = 0;
  {
    array[batch_size, block_size] vector[n_embed] x_validation;
    for (b in 1:batch_size) {
      for (t in 1:block_size) {
	x_validation[b, t] = token_embedding[xb_val[b, t]] + position_embedding[t];
      }
    }
    array[batch_size, block_size] vector[head_size] x_validation_self_attention;
    x_validation_self_attention = self_attention(x_validation, key, query, value);

    for (b in 1:batch_size) {
      for (t in 1:block_size) {
	vector[vocab_size] logits;
	logits = lm_head(x_validation_self_attention[b, t],
			 lm_head_multiplier, lm_head_offset);
	loss_validation += categorical_logit_lpmf(yb_val[b, t] | logits);
      }
    }
    loss_validation /= batch_size * block_size;
  }
  print("************************************************************");
  print("train loss ", -loss, ", val loss ", -loss_validation);
  print("************************************************************");


  array[max_new_tokens] int<lower = 1, upper = vocab_size> new_tokens;
  new_tokens[1] = 1;
  {
    array[block_size] vector[n_embed] x_new = rep_array(rep_vector(0, n_embed), block_size);
    array[block_size] vector[head_size] x_new_self_attention;

    for (n in 2:(block_size + 1)) {
      x_new[n - 1] = token_embedding[new_tokens[n - 1]] + position_embedding[n - 1];
      x_new_self_attention = self_attention(x_new, key, query, value);
      new_tokens[n] = categorical_logit_rng(lm_head(x_new_self_attention[n - 1],
						    lm_head_multiplier,
						    lm_head_offset));
    }
    for (n in block_size + 2:max_new_tokens) {
      for (t in 1:block_size) {
	x_new[t] = token_embedding[new_tokens[(n - block_size - 1) + t]] + position_embedding[t];
      }
      x_new_self_attention = self_attention(x_new, key, query, value);
      new_tokens[n] = categorical_logit_rng(lm_head(x_new_self_attention[block_size],
						    lm_head_multiplier,
						    lm_head_offset));
    }
  }
}

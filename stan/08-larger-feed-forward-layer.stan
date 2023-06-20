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

  // multi head self attention
  array[,] vector multi_head_self_attention(array[,] vector x,        // batch_size, block_size, n_embed
					    array[] matrix key,       // n_heads, n_embed x head_size
					    array[] matrix query,     // n_heads, n_embed x head_size
					    array[] matrix value) {   // n_heads, n_embed x head_size
    // -> n_embed, block_size, head_size
    int batch_size = dims(x)[1];
    int block_size = dims(x)[2];
    int n_head = dims(key)[1];
    int n_embed = rows(key[1]);
    int head_size = cols(key[1]);

    array[batch_size, block_size] vector[n_embed] out;
    for (b in 1: batch_size) {
      for (n in 1:n_head) {
	array[block_size] vector[head_size] x_self = self_attention(x[b], key[n], query[n], value[n]);
	for (t in 1:block_size) {
	  out[b, t, (((n-1) * head_size) + 1):(n * head_size)] = x_self[t];
	}
      }
    }
    return out;
  }

  // Rectified linear unit. Activation function.
  vector ReLU(vector x) {
    vector[rows(x)] y;
    for (n in 1:rows(x)) {
      y[n] = fmax(0.0, x[n]);
    }
    return y;
  }
  
}
data {
  int<lower = 1> vocab_size;           // e.g. 65
  int<lower = 1> batch_size;           // e.g. 32
  int<lower = 1> block_size;           // e.g. 8
  int<lower = 1> n_embed;              // e.g. 32
  int<lower = 1> n_head;               // e.g. 2

  array[batch_size, block_size] int<lower = 1, upper = vocab_size> xb;
  array[batch_size, block_size] int<lower = 1, upper = vocab_size> yb;

  array[batch_size, block_size] int<lower = 1, upper = vocab_size> xb_val;
  array[batch_size, block_size] int<lower = 1, upper = vocab_size> yb_val;
  
  int<lower = 0> max_new_tokens;
}
transformed data {
  int head_size = n_embed %/% n_head;
}
parameters {
  array[vocab_size] vector[n_embed] token_embedding;        // 02 - change in token_embedding size
  matrix[vocab_size, n_embed] lm_head_multiplier;           // 02 - new parameters
  vector[vocab_size] lm_head_offset;                        // 02 - new parameters

  array[block_size] vector[n_embed] position_embedding;     // 03 - new parameter

  // Multi-head self attention
  array[n_head] matrix[n_embed, head_size] key;             // 05 - update for multi head
  array[n_head] matrix[n_embed, head_size] query;           // 05 - update for multi head
  array[n_head] matrix[n_embed, head_size] value;           // 05 - update for multi head

  // Feed forward
  matrix[4 * n_embed, n_embed] feed_forward_multiplier;      // 07 - update for larger feed forward
  vector[4 * n_embed] feed_forward_offset;                   // 07 - update for larger feed forward

  matrix[n_embed, 4 * n_embed] feed_forward_proj_multiplier; // 07 - new parameter
  vector[n_embed] feed_forward_proj_offset;                  // 07 - new parameter
}
transformed parameters {
  array[batch_size, block_size] vector[n_embed] x;
  for (b in 1:batch_size) {
    for (t in 1:block_size) {
      x[b, t] = token_embedding[xb[b, t]] + position_embedding[t];
    }
  }
  array[batch_size, block_size] vector[n_embed] x_self_attention = multi_head_self_attention(x, key, query, value);
  // 07 - skip connection
  for (b in 1:batch_size) {
    for (t in 1:block_size) {
      x[b, t] += x_self_attention[b, t];
    }
  }

  for (b in 1:batch_size) {
    for (t in 1:block_size) {
      // 07 - skip connection
      x[b, t] += feed_forward_proj_multiplier
	* ReLU(feed_forward_multiplier * x[b, t] + feed_forward_offset)
	+ feed_forward_proj_offset;
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
  real loss_validation = 0;
  {
    array[batch_size, block_size] vector[n_embed] x_val;
    for (b in 1:batch_size) {
      for (t in 1:block_size) {
	x_val[b, t] = token_embedding[xb_val[b, t]] + position_embedding[t];
      }
    }
    
    array[batch_size, block_size] vector[n_embed] x_val_self_attention = multi_head_self_attention(x_val, key, query, value);
    for (b in 1:batch_size) {
      for (t in 1:block_size) {
	x_val[b, t] += x_val_self_attention[b, t];
      }
    }
    
    for (b in 1:batch_size) {
      for (t in 1:block_size) {
	x_val[b, t] += feed_forward_proj_multiplier
	  * ReLU(feed_forward_multiplier * x_val[b, t] + feed_forward_offset)
	  + feed_forward_proj_offset;
	
	vector[vocab_size] logits = lm_head(x_val[b, t], lm_head_multiplier, lm_head_offset);
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
    array[1, block_size] vector[n_embed] x_new;
    array[1, block_size] vector[head_size] x_new_self_attention;

    for (n in 2:max_new_tokens) {
      x_new = rep_array(rep_vector(0, n_embed), 1, block_size);
      for (t in 1:min(n - 1, block_size)) {
	x_new[1, t] = token_embedding[new_tokens[max(0, n - 1 - block_size) + t]] + position_embedding[t];
      }
      
      x_new_self_attention = multi_head_self_attention(x_new, key, query, value);
      for (t in 1:min(n - 1, block_size)) {
	x_new[1, t] += x_new_self_attention[1, t];
      }

      int idx = min(n - 1, block_size);
      x_new[1, idx] += feed_forward_proj_multiplier
	* ReLU(feed_forward_multiplier * x_new[1, idx] + feed_forward_offset)
	+ feed_forward_proj_offset;
      new_tokens[n] = categorical_logit_rng(lm_head(x_new[1, idx],
						    lm_head_multiplier,
						    lm_head_offset));
    }
  }
}

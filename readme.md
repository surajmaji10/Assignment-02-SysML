# Simple Transformer Model From Scratch in Python

## Overview
This program implements fundamental components of a Transformer model, including input embedding generation, positional encoding, self-attention, multi-head attention, matrix operations, and output probability computation. The script generates random embeddings, applies attention mechanisms, and computes probabilities using a softmax function.

## Functions

### 1. `generate_random_values(n, d_model)`
Generates a matrix of size `n x d_model` with random values between 0 and 1.

### 2. `save_matrix_m_n(matrix, m, n, filename)`
Saves a given `m x n` matrix into a file.

### 3. `save_embedding_n(input_embeddings_n, n, d_model, filename)`
Saves the input embedding matrix into a file.

### 4. `print_embedding_n(input_embeddings_n, n, d_model)`
Prints the input embeddings to the console.

### 5. `vector_add(vector_x, vector_y, size)`
Performs element-wise addition of two vectors of length `size`.

### 6. `PE(pos, idx)`
Computes the positional encoding value for a given position and index using sine and cosine functions.

### 7. `positional_encoder_vector(vector, size, pos)`
Applies positional encoding to a given vector.

### 8. `positional_encoder_n(input_embeddings_n, n, d_model)`
Applies positional encoding to an entire sequence of embeddings.

### 9. `matrix_transpose(matrix, m, n)`
Computes the transpose of an `m x n` matrix.

### 10. `print_matrix(matrix, n_rows, head_width)`
Prints a matrix in a readable format.

### 11. `concat_heads(heads_list, n_heads, n, head_width)`
Concatenates `n_heads` number of attention heads into a single matrix.

### 12. `scalar_matrix_mul(A, m, n, alpha)`
Multiplies every element in an `m x n` matrix `A` by a scalar `alpha`.

### 13. `matrix_add(A, B, m, n)`
Performs element-wise addition of two `m x n` matrices.

### 14. `matrix_mul(A, B, p, q, r, s)`
Performs matrix multiplication between two matrices `A (p x q)` and `B (r x s)`.

### 15. `normalize(vector, n)`
Normalizes a vector by computing its mean and standard deviation.

### 16. `matrix_normalize(A, m, n)`
Applies normalization to each row in an `m x n` matrix.

### 17. `split_into_heads(A, n, d_model, n_heads)`
Splits an `n x d_model` matrix into `n_heads` separate matrices of equal width.

### 18. `soft_max(M, n, d)`
Computes the softmax function row-wise for an `n x d` matrix.

### 19. `self_attention(Q, K, V, n, d_k)`
Computes scaled dot-product self-attention given query `Q`, key `K`, and value `V` matrices.

### 20. `multi_head_attention(Q, K, V, n, d_model, n_heads)`
Computes multi-head attention by applying self-attention across multiple attention heads.

### 21. `matrix_add_broadcasted(A, b, p, q, r, s)`
Performs row-wise broadcasted addition of a matrix `A` and a vector `b`.

### 22. `linear_transformation_with_weight_bias(X, W, b, n, d_model, wd_model1, wd_model2, b1, vocab_size)`
Applies a linear transformation followed by bias addition to the input matrix.

## Execution Flow

## Author
### @Akash Maji
### @akashmaji@iisc.ac.in




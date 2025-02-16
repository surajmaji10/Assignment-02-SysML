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

## Execution Flow Summary
This script implements a basic Transformer without training, but the logic follows the original Transformer paper (Attention Is All You Need).
- Generate input word embeddings.
- Apply positional encoding (to provide sequence order).
- Encoder:
  - Perform multi-head self-attention. 
  - Apply residual connections and normalization.
  - Produce the encoder output.
- Decoder:
  - Generate embeddings & apply positional encoding.
  - Perform masked self-attention (no future words).
  - Attend to encoder outputs using encoder-decoder attention.
  - Apply residual connections and normalization.
- Final Prediction:
  - Pass through fully connected layer.
  - Apply softmax to get probabilities.
  - Save and print results.

# Transformer Execution Flow

## 1. Define Parameters
The script starts by defining key hyperparameters:

- `n = 256`: Number of words (sequence length).
- `d_model = 256`: Embedding size (each word is represented as a 256-dimensional vector).
- `n_heads = 8`: Number of attention heads for Multi-Head Attention.
- `vocab_size = 1024`: Vocabulary size (number of unique words).

## 2. Generate Random Embeddings
```python
input_embeddings_n = generate_random_values(n, d_model)
output_embeddings_n = generate_random_values(n, d_model)
```
Generates a matrix of shape (256, 256) filled with random values to represent word embeddings.
These embeddings are used for both input (encoder) and output (decoder) sequences.

## 3. Positional Encoding
Since Transformers do not have built-in sequential order (unlike RNNs), positional encoding is used to inject sequence information.

```python
input_encoding_n = positional_encoder_n(input_embeddings_n, n, d_model)
```
- `PE(pos, idx)`: Computes sine and cosine functions for different positions.
- `positional_encoder_n(input_embeddings_n, n, d_model)`: Applies positional encoding to input embeddings.

## 4. Encoder Multi-Head Attention
The encoder applies multi-head self-attention to process input sequences.

```python
mha_result = multi_head_attention(input_encoding_n, input_encoding_n, input_encoding_n, n, d_model)
```
- The **query (Q), key (K), and value (V)** are all the **same** as `input_encoding_n`.
- `split_into_heads(Q, n, d_model, n_heads)` splits the input into 8 attention heads.
- `self_attention(Q, K, V, n, d_k)` computes attention scores.
- The results from all heads are concatenated and passed through a linear transformation.

## 5. Add & Normalize (Residual Connection)
```python
added_output = matrix_add(mha_result, input_encoding_n, n, d_model)
normalized_output = matrix_normalize(added_output, n, d_model)
encoder_output = normalized_output
```
- The **self-attention output** is **added back** to the input (residual connection).
- The result is **normalized** for stable training.

`encoder_output` now represents the final **processed input sequence**.

## 6. Decoder Input Preparation
- Generate random **decoder embeddings** (`output_embeddings_n`).
- Apply **positional encoding**.

```python
output_encoding_n = positional_encoder_n(input_embeddings_n, n, d_model)
```

## 7. Decoder Self-Attention (Masked)
```python
masked_mha_result = self_attention(output_encoding_n, output_encoding_n, output_encoding_n, n, d_model)
```
- The decoder **does not** look at future words (hence, "masked" self-attention).
- Only previous words contribute to predictions.

## 8. Add & Normalize (Decoder Self-Attention Output)
```python
added_output = matrix_add(masked_mha_result, output_encoding_n, n, d_model)
normalized_output = matrix_normalize(added_output, n, d_model)
```
- Adds residual connection.
- Applies **layer normalization**.

## 9. Decoder Multi-Head Attention (Encoder-Decoder Attention)
```python
mha_result = multi_head_attention(encoder_output, encoder_output, normalized_output, n, d_model)
```
- The decoder attends to **encoder outputs**.
- The encoder **acts as the key and value**, while the decoder's self-attention output is the **query**.

## 10. Add & Normalize (Encoder-Decoder Attention Output)
```python
added_output = matrix_add(mha_result, normalized_output, n, d_model)
normalized_output = matrix_normalize(added_output, n, d_model)
```
- Another **residual connection**.
- Another **layer normalization**.

## 11. Fully Connected (Final Prediction Layer)
```python
weights_stabilized = generate_random_values(d_model, vocab_size)
bias_vector = generate_random_values(1, vocab_size)[0]

linear_output = linear_transformation_with_weight_bias(
    normalized_output, weights_stabilized, bias_vector, n, d_model, d_model, vocab_size, 1, vocab_size
)
```
- Applies a **linear transformation** to map `d_model=256` features to `vocab_size=1024` classes (word probabilities).

## 12. Softmax (Convert Scores to Probabilities)
```python
softmax_output = soft_max(linear_output, n, vocab_size)
output_probabilities = softmax_output
```
- Converts raw scores into **probabilities** using **softmax**.

## 13. Save Outputs to Files
At various stages, matrices are **printed and saved** to files:

```python
save_matrix_m_n(output_probabilities, n, vocab_size, "probs_output.txt")
```
- This allows visualization of Transformer computations.

## Final Output
- `output_probabilities`: A **(256, 1024) matrix** containing probabilities for each token in the sequence.
- The **highest probability word** in each position is the predicted word.



## Author
### @Akash Maji
### @akashmaji@iisc.ac.in




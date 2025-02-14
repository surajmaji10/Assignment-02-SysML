import random as rd
import math
# embedding size = 256 (i.e. d_model = 256)
d_model = 256
TEN_K = 10000
n_heads = 8

vocab_size = 1024

# to store input embeddings
input_embeddings_n = []
# number of words
n = 256
# to store output probabilities
output_probabilities = []

# generate random embedding for each word
def generate_random_values(n, d_model):
    matrix = []
    for i in range(n):
        row = [rd.random() for j in range(d_model)]
        matrix.append(row)
    return matrix


input_embeddings_n = generate_random_values(n, d_model)

weights_learned = generate_random_values(d_model, vocab_size)
bias_vector = generate_random_values(1, vocab_size)[0]

def save_matrix_m_n(matrix, m, n, filename="sample.txt"):
    with open(filename, "w") as f:
        for i in range(m):
            row = matrix[i]
            assert len(row) == n
            for j in range(n):
                f.write(f"{row[j]:8.4f}  ")  
            f.write("\n")  
        f.write("\n")

def save_embedding_n(input_embeddings_n, n, d_model, filename="embedding_output.txt"):
    with open(filename, "w") as f:
        for i in range(n):
            embedding = input_embeddings_n[i]
            assert len(embedding) == d_model
            for j in range(len(embedding)):
                f.write(f"{embedding[j]:8.4f}  ")  
            f.write("\n")  
        f.write("\n")  

# print and see the embedding
def print_embedding_n(input_embeddings_n, n, d_model):
    for i in range(n):
        embedding = input_embeddings_n[i]
        assert len(embedding) == d_model
        for j in range(len(embedding)):
            print(f"{embedding[j]:8.4f}", " ", end="")
        print("")
    print("")

# print_embedding_n(input_embeddings_n, n, d_model)

# does simple vector addition of `vector_x` and `vector_y` each of size `size`
def vector_add(vector_x, vector_y, size):
    assert len(vector_x) == len(vector_y) and size == len(vector_y)
    vector_z = []
    for i in range(size):
        vector_z.append(vector_x[i] + vector_y[i])
    return vector_z


# do positional encoding using `sin` or `cos`
def PE(pos, idx):
    # use sine()
    if idx % 2 == 0:
        return math.sin(pos/(TEN_K ** (idx/d_model)))
    else:
        return math.cos(pos/(TEN_K ** (idx/d_model)))

    
# encode a vector based on position in sequence
def positional_encoder_vector(vector, size, pos):
    encoded_vector = []
    if size is None:
        size = len(vector)
    for j in range(size):
        encoded = PE(pos, j)
        encoded_vector.append(encoded)
    return encoded_vector

# positionally encode a sequence
def positional_encoder_n(input_embeddings_n, n, d_model):
    assert len(input_embeddings_n) == n
    for pos in range(n):
        vector = input_embeddings_n[pos]
        assert len(vector) == d_model
        pos_encoded_vector = positional_encoder_vector(vector, d_model, pos)
        input_embeddings_n[pos] = vector_add(vector, pos_encoded_vector, d_model)
    return input_embeddings_n

positional_encoder_n(input_embeddings_n, n, d_model)
# print_embedding_n(input_embeddings_n, n, d_model)
save_embedding_n(input_embeddings_n, n, d_model, "saved.txt")

# do matrix transpose
def matrix_transpose(matrix, m, n):
    assert len(matrix) == m
    transposed = []
    for i in range(n):
        transposed.append([0.0 for j in range(m)])
    for i in range(m):
        for j in range(n):
            transposed[j][i] = matrix[i][j]
    return transposed

# print(matrix_transpose(matrix=[[1, 2, 3], [4, 5, 6]], m = 2, n = 3))


def print_matrix(matrix, n_rows, head_width):
    assert n_rows == len(matrix)
    for i in range(n_rows):
        n_cols = len(matrix[i])
        assert n_cols == head_width
        for j in range(n_cols):
            print(f"{matrix[i][j]:8.4f}", " ", end="")
        print()
    print()


heads_list = [
    [
        [1, 2],
        [3, 4],
        [4, 5]
    ],
    [   [7, 8],
        [9, 10],
        [11, 12]
    ],
    [
        [1, 2],
        [3, 4],
        [4, 5]
    ],
    [   [7, 8],
        [9, 10],
        [11, 12]
    ]

]

def concat_heads(heads_list, n_heads, n, head_width):
    assert len(heads_list) == n_heads
    n_rows = len(heads_list[0])
    for head in heads_list:
        assert len(head) == n_rows
        print_matrix(head, n_rows, head_width)
    # multi_head = [[0.0 for _ in range(n_heads * head_width)] for row in range(n_rows)]
    multi_head = []
    for i in range(n_rows):
        row = []
        for j in range(n_heads):
            row.extend(heads_list[j][i])
        multi_head.append(row)

    return multi_head
    
    
# mh = concat_heads(heads_list, 4, 3, 2)
# print_matrix(mh, 3, 8)
    
# implement scalar matrix multiplication
def scalar_matrix_mul(A, m, n, alpha):
    assert len(A) == m
    B = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        assert len(A[i]) == n
        for j in range(n):
            B[i][j] = A[i][j]
            B[i][j] *= alpha
    return B

# do matrix matrix addition
def matrix_add(A, B, m, n):
    assert len(A) == m and len(B) == m
    C = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        assert len(A[i]) == n and len(B[i]) == n
        for j in range(n):
            C[i][j] += A[i][j] + B[i][j]
    return C

# implement matrix multiplication
def matrix_mul(A, B, p, q, r, s):
    assert q == r
    C = [[0.0 for col in range(s)] for row in range(p)]
    for i in range(p):
        for j in range(s):
            C[i][j] = 0.0
            for k in range(r):
                C[i][j] = C[i][j] + A[i][k] * B[k][j]
    return C

# p, q, r, s = 3, 4, 4, 5
# # Generate random p.q matrix A with values 0 or 1
# A = [[rd.random() for _ in range(q)] for _ in range(p)]

# # Generate random r.s matrix B with values 0 or 1
# B = [[rd.random() for _ in range(s)] for _ in range(r)]

# C = matrix_mul(A, B, p, q, r, s)
# print_matrix(C, p, s)

# normalize a vector
def normalize(vector, n):
    assert len(vector) == n
    mean = sum(vector)/len(vector)
    squared_distances = [(x - mean)**2 for x in vector]
    variance = sum(squared_distances)
    stddev = math.sqrt(variance)

    normalized = []
    for x in vector:
        y = (x - mean)/stddev
        normalized.append(y)
    return normalized

# define matrix normalize
def matrix_normalize(A, m, n):
    assert len(A) == m
    C = []
    for row in A:
        assert len(row) == n
        C.append(normalize(row, n))
    return C

# 1 2 3 4 5 6 7 8
# 2 3 4 6 7 8 1 1
# 2 3 4 5 6 7 8 9



## split a matrix into heads
def split_into_heads(A, n, d_model, n_heads):
    assert len(A) == n
    head_width = d_model // n_heads
    for row in A:
        assert len(row) == d_model
    A_splitted = [[[0.0 for _ in range(head_width)] for _ in range(n)] for _ in range(n_heads)]
    for i in range(n):
        for j in range(d_model):
            head_idx = j // head_width
            k = j % head_width
            A_splitted[head_idx][i][k] = A[i][j]
    return A_splitted



## define softmax logic
def soft_max(M, n, d):
    # Matrix is n * d
    softmax_M = []
    assert len(M) == n

    for i in range(n):
        row = M[i]
        assert len(row) == d
        max_val = max(row)  # For numerical stability
        exp_row = [math.e**(x - max_val) for x in row]
        sum_exp = sum(exp_row)
        softmax_row = [x / sum_exp for x in exp_row]
        softmax_M.append(softmax_row)
    return softmax_M

## define self-attention logic
def self_attention(Q, K, V, n, d_k):
    assert len(Q) == n and len(K) == n and len(V) == n
    for i in range(n):    
        assert len(Q[i]) == d_k and len(K[i]) == d_k and len(V[i]) == d_k

    # K.Qt
    Q_t = matrix_transpose(Q, n, d_k)
    upper = matrix_mul(K, Q_t, n, d_k, d_k, n)
    upper = scalar_matrix_mul(upper, n, n, 1 / (math.sqrt(d_k)))
    first = soft_max(upper, n, n)
    attention = matrix_mul(first, V, n, n, n, d_k)
    return attention

## define multi-head attention
def multi_head_attention(Q, K, V, n, d_model):
    W_q = generate_random_values(d_model, d_model)
    W_k = generate_random_values(d_model, d_model)
    W_v = generate_random_values(d_model, d_model)
    W_o = generate_random_values(d_model, d_model)

    Q_ = matrix_mul(Q, W_q, n, d_model, d_model, d_model)
    K_ = matrix_mul(K, W_k, n, d_model, d_model, d_model)
    V_ = matrix_mul(V, W_v, n, d_model, d_model, d_model)

    d_k = d_model//n_heads
    Q_list = split_into_heads(Q_, n, d_model, n_heads) # d_k = d_model//n_heads
    K_list = split_into_heads(K_, n, d_model, n_heads) # n_heads * n * (d_k)
    V_list = split_into_heads(V_, n, d_model, n_heads) # n_heads * n * (d_model//n_heads)

    heads_list = []
    for h in range(n_heads):
        head_h = self_attention(Q_list[h], K_list[h], V_list[h], n, d_k)
        heads_list.append(head_h)
    # return heads_list

    heads_merged = concat_heads(heads_list, n_heads, n, d_k) # n * (n_heads*d_k) = n * d_model
    mh_attenention = matrix_mul(heads_merged, W_o, n, d_model, d_model, d_model)
    return mh_attenention


def matrix_add_broadcasted(A, b, p, q, r, s):
    assert r == 1 and q == s and len(A) == p and len(b) == s
    added_matrix = []
    for row in A:
        added_row = vector_add(row, b, q)
        added_matrix.append(added_row)
    return added_matrix

# define linear transformation
def linear_transformation_with_weight_bias(X, W, b, n, d_model, wd_model1, wd_model2, b1, vocab_size):
    assert wd_model1 == d_model and wd_model2 == vocab_size and b1 == 1
    Y = matrix_mul(X, W, n, d_model, wd_model1, wd_model2) # n * vocab_size
    Z = matrix_add_broadcasted(Y, b, n, vocab_size, 1, vocab_size) 
    return Z




# res = self_attention(input_embeddings_n, input_embeddings_n, input_embeddings_n, n, d_model)
# print_matrix(res, n, d_model)
# print_matrix(input_embeddings_n, n, d_model)

# added_output = matrix_add(res, input_embeddings_n, n, d_model)
# print_matrix(added_output, n, d_model)


# normalized_output = matrix_normalize(added_output, n, d_model)
# print_matrix(normalized_output, n, d_model)

# matrix = [
#     [1, 2, 3, 4, 5, 6, 7, 8],
#     [2, 3, 4, 6, 7, 8, 1, 1],
#     [2, 3, 4, 5, 6, 7, 8, 9]
# ]

# out = split_into_heads(matrix, 3, 8, 4)
# print(out)


res = multi_head_attention(input_embeddings_n, input_embeddings_n, input_embeddings_n, n, d_model)
print_matrix(res, n, d_model)
print_matrix(input_embeddings_n, n, d_model)

added_output = matrix_add(res, input_embeddings_n, n, d_model)
print_matrix(added_output, n, d_model)


normalized_output = matrix_normalize(added_output, n, d_model)
print_matrix(normalized_output, n, d_model)

output_embdeddings = generate_random_values(n, d_model)
weights_stabilized = generate_random_values(d_model, vocab_size)
bias_vector = generate_random_values(1, vocab_size)[0]
output_probabilities = linear_transformation_with_weight_bias(output_embdeddings, weights_stabilized, bias_vector, \
                                       n, d_model, d_model, vocab_size, 1, vocab_size)
print_matrix(output_probabilities, n, vocab_size)


soft = soft_max(output_probabilities, n, vocab_size)
print_matrix(soft, n, vocab_size)
save_matrix_m_n(soft, n, vocab_size, "tmp.txt")



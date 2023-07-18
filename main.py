import torch
import numpy

# read data from dataset
with open('assets/fairytale_dataset.txt', 'r', encoding='utf-8') as dataset:
    data = dataset.read()


vocabulary = sorted(list(set(data)))
# print('Vocabulary: ', vocabulary)
# print(len(vocabulary))


# map characters to integers
def encode(string):
    encoded = []
    for char in string:
        encoded.append(ord(char))
    return encoded


# maps a list of integers to characters
def decode(int_list):
    decoded = []
    for integer in int_list:
        decoded.append(chr(integer))
    return ''.join(decoded)


print(encode('izu mma'))
print(decode([105, 122, 117, 32, 109, 109, 97]))

data_tensor = torch.tensor(encode(data), dtype=torch.long)
# print(data_tensor.shape, data_tensor.dtype)
print(data_tensor[:1000])

print(decode(data_tensor[:1000]))
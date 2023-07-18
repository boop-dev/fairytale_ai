import torch
import numpy

# read data from dataset
with open('assets/fairytale_dataset.txt', 'r', encoding='utf-8') as dataset:
    data = dataset.read()

vocabulary = sorted(list(set(data)))


# print('Vocabulary: ', vocabulary)
# print(len(vocabulary))


# map characters to integers using ASCII
def encode(string):
    encoded = []
    for char in string:
        encoded.append(ord(char))
    return encoded


# maps a list of integers to characters using ASCII
def decode(int_list):
    decoded = []
    for integer in int_list:
        decoded.append(chr(integer))
    return ''.join(decoded)


print(encode('izu mma'))
print(decode([105, 122, 117, 32, 109, 109, 97]))

data_tensor = torch.tensor(encode(data), dtype=torch.long)
print(data_tensor.shape, data_tensor.dtype)
print(data_tensor[:1000])

print(decode(data_tensor[:1000]))

# divide the data into training and validation sets
divider = int(0.9 * len(data))
training_data = data_tensor[:divider]
validation_data = data_tensor[divider:]

# setting up context
'''
    context_length: this is the maximum amount of data that can be used by the transformer to predict the next value
    in one instance

    Each chunk of data to be fed into the transformer is context_length + 1 units long. 
    Say the chunk is [ 1, 2, 3, 4, 5, 6 ], the first context is [1] and the transformer has to predict the next number
    based on that context. The next context is [ 1, 2 ] and the transformer has to predict the next number based on 
    [ 1, 2 ]. This goes on until the context is [ 1, 2, 3, 4, 5 ] (i.e the length of the context only goes up to the  
    context_length). 
'''
context_length = 8
x = training_data[:context_length]
y = training_data[1:context_length + 1]
for i in range(context_length):
    context = x[:i + 1]
    target = y[i]

# batch dimensioning
'''
    This part determines how many instances of the data will be fed into the transformer at a time.
    x and y are like 2d arrays
    
    batch_size: how many instances will be processed in parallel
    batch_data: is either the training or validation data depending on which one is being used
    ix: is a tensor(of length batch_size) of randomly generated numbers from 0 to len(batch_data)
    x: is a tensor of all the context tensors
    y: is a tensor of all the target tensors
    
    A sample run will give :
            x.shape = torch.Size([4, 8])
            x = tensor([[ 59,  32,  97, 110, 100,  32,  97, 102],
                [ 32, 115,  99, 111, 110,  99, 101, 115],
                [101, 102, 117, 115, 101, 100,  32, 116],
                [ 97, 100,  32,  98, 101, 101, 110,  32]])
                
            y.shape = torch.Size([4, 8])
            y = tensor([[ 32,  97, 110, 100,  32,  97, 102, 116],
                    [115,  99, 111, 110,  99, 101, 115,  32],
                    [102, 117, 115, 101, 100,  32, 116, 111],
                    [100,  32,  98, 101, 101, 110,  32, 101]])
'''
batch_size = 4


def get_batch(split):
    if split == 'training':
        batch_data = training_data
    else:
        batch_data = validation_data

    ix = torch.randint(len(batch_data) - context_length, (batch_size, ))
    x = torch.stack([batch_data[num: num + context_length] for num in ix])
    y = torch.stack([batch_data[num + 1: num + context_length + 1] for num in ix])
    return x, y


xb, yb = get_batch('training')
print(xb.shape)
print(xb)
print(yb.shape)
print(yb)


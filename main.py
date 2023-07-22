import torch
import numpy as np

# read data from dataset
with open('assets/fairytale_dataset.txt', 'r', encoding='utf-8') as dataset:
    data = dataset.read()

vocabulary = sorted(list(set(data)))


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


char_to_int = dict((char, ord(char)) for char in vocabulary)
int_to_char = dict((ord(char), char) for char in vocabulary)

data_tensor = torch.tensor(encode(data), dtype=torch.long)

# print(decode(data_tensor[:1000]))

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
context_length = 5
x = training_data[:context_length]
y = training_data[1:context_length + 1]
# for i in range(context_length):
#     context = x[:i + 1]
#     target = y[i]
# print('ccc', context)
# print('ttt', target)

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

    # generate random indexes from 0 to length of batch_data - context_size
    # subtracting context_length because if we get an index where the remaining characters are not up to the
    # context_length then the neural net cannot predict the next character
    rand_indexes = torch.randint(len(batch_data) - context_length, (batch_size,))

    # generate contexts and targets based on the random indexes from above
    context_stack = torch.stack([batch_data[num: num + context_length] for num in rand_indexes])
    target_stack = torch.stack([batch_data[num + 1: num + context_length + 1] for num in rand_indexes])
    return context_stack, target_stack


# xb, yb = get_batch('training')
# print(xb.shape)
# print(xb)
# print(yb.shape)
# print(yb)

# make 2000 training batches
contexts = []
for i in range(10):
    contexts.append(get_batch('training'))


# make numpy arrays filled with zeroes for the contexts and targets
def make_np_array(batch_length):
    contexts_np_array = np.zeros((batch_length, context_length, len(vocabulary)), dtype=np.bool_)
    targets_np_array = np.zeros((batch_length, context_length, len(vocabulary)), dtype=np.bool_)
    return contexts_np_array, targets_np_array


# populate the arrays
'''
    batch: is a torch.stack of two 2D tensors. They are contexts and targets tensors respectively.
    contexts_array, targets_array: are two numpy zeros arrays that are populated with True or False based on the 
                                   tensors in the batch. They each contain a 3D matrix of all possible values 
                                   that a character could be. The populate function updates the corresponding
                                   positions in these matrices based on the values in the batch tensors.
'''


def populate(batch):
    contexts_array, targets_array = make_np_array(len(batch[0]))
    keys = list(int_to_char.keys())
    torch.set_printoptions(profile='full')

    # count_a, count_b and index will track the 1st, 2nd and 3rd dimensions of the matrix respectively
    count_a = 0
    for context in batch[0]:
        count_b = 0
        for integer in context:
            index = keys.index(integer)
            contexts_array[count_a][count_b][index] = 1
            count_b += 1
        count_a += 1

    # reset local variables and do the same for the targets
    count_a = 0
    for target in batch[1]:
        count_b = 0
        for integer in target:
            index = keys.index(integer)
            targets_array[count_a][count_b][index] = 1
            count_b += 1
        count_a += 1

    return contexts_array, targets_array


print('test')
torch.set_printoptions(threshold=10_000)
print(populate(get_batch('training'))[0][0])




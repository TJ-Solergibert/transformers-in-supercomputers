from math import ceil

def shift_bit_length(x):
    # Get greater power of 2 number: 500.7 --> 512, 1000.2 --> 1024
    return 1<<(ceil(x)-1).bit_length()
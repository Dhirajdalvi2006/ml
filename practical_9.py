# Practical 9: McCulloch-Pitts ANDNOT Neuron
def and_not(x1, x2):
    return int(x1 - x2 >= 1)       # weights: 1, -1; threshold: 1

print('X1 | X2 | Output')
print('-' * 16)
for x1 in (0, 1):
    for x2 in (0, 1):
        print(f' {x1} | {x2}  |   {and_not(x1, x2)}')

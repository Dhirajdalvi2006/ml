def mp(x1, x2):
    return 1 if x1 - x2 >= 1 else 0

print("X1 X2 Output")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(x1, x2, mp(x1, x2))
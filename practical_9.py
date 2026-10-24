# Practical 9: McCulloch-Pitts Neural Net for ANDNOT Function
# Implements a McCulloch-Pitts neuron to compute the ANDNOT logical operation.

def mcculloch_pitts_andnot(x1, x2):
    # Weights: w1 = 1 for x1, w2 = -1 for x2 (inhibitory input)
    w1, w2 = 1, -1
    threshold = 1

    # Weighted sum
    total_input = (x1 * w1) + (x2 * w2)

    # Threshold activation function
    return 1 if total_input >= threshold else 0

# Test all combinations for ANDNOT (x1 AND NOT x2)
print("McCulloch-Pitts Neural Net: ANDNOT Truth Table")
print(" X1 | X2 | Output ")
print("-" * 18)
for x1 in [0, 1]:
    for x2 in [0, 1]:
        out = mcculloch_pitts_andnot(x1, x2)
        print(f"  {x1} |  {x2} |   {out}   ")

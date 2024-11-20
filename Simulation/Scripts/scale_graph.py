import numpy as np
import matplotlib.pyplot as plt

RMS = [0.0337911, 0.00844625, 0.00706629, 0.00579778, 0.00623420, 0.00690941, 0.00765464, 0.00869721]
Max = [0.0720821, 0.0163713, 0.0173434, 0.0201790, 0.0224573, 0.0253414, 0.0282848, 0.0314431]

plt.plot(range(1, len(RMS) + 1),RMS, label="RMS")
plt.plot(range(1, len(Max) + 1), Max, label="Max Error")
plt.title("RMS and Max Error vs. # times scaled")
plt.legend("RMS", "Max Error")
plt.xlabel("Number of times scaled")
plt.show()

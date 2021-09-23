import os
import pickle
import numpy as np

data = np.load("../spect/p111/p111_001.npy")
print(data)


# metadata = pickle.load(open('metadata.pkl', "rb"))
# for i in metadata:
# 	np.save(i[0], i[1])


# spect_vc = pickle.load(open('results.pkl', "rb"))
# data = np.load("../conversion/p925_001xp925.npy")

# print(spect_vc[0][1].shape)
# print(data.shape)

# print(sum(spect_vc[0][1] != data))
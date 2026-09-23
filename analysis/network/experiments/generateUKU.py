import numpy as np
import pickle
import sys
from scipy.io import savemat

name = sys.argv[1]
myDict = pickle.load(open(name,'rb'))
components = myDict['components']
comp2 = components**2
norm = np.sum(comp2,axis=0)
print(norm.shape)

comp2_norm = comp2/norm

out_dict = {}
out_dict['components_supervised'] = components[0]
out_dict['components_unsupervised'] = components[1:]
out_dict['ukuNorm'] = comp2_norm

savemat('Aggression_elasticNet20220814.mat',out_dict)


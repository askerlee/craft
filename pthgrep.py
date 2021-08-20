import torch
import sys

a=torch.load(sys.argv[1], map_location='cpu')
keys = sys.argv[2].split(",")
for var in a.keys():
    for key in keys:
        if key in var:
            print("{} ({}) => {}".format(var, key, a[var]))
            
        
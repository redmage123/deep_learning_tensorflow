#!/usr/bin/env python3
import numpy as np

# Create an array 'a' as a 2 by 2 dimensional array initialized to zeros. 
a = np.zeros((2,2))

# Create an array 'b' as a 2 by 2 dimensional array initialized to ones. 
b = np.ones((2,2))

# Add the two up.  The axis parameter refers to columsn vs. rows.  Axis=0
# refers aggregation along the row, axis=1 refers to aggregatio along
# the columns. 
print (np.sum (b,axis=1))

# The reshape method changes a to be a 1 X 4 array. 
print (np.reshape(a,(1,4)))

import pandas as pd

def intersection(a, b):
  '''
  Takes the y point from the first argument, for now.
  '''
  distance = (a - b).abs()
  mindex = distance.idxmin()
  return (mindex, a[mindex])

def quantize(x, step):
  sign = pd.apply(np.sign, x) 
  quantized = pd.apply(np.floor, (value.abs() / step) + 0.5)
  return (sign * quantized)

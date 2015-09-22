import numpy as np

def cartesian(arrays, out=None):
  arrays = [np.asarray(x) for x in arrays]
  dtype = arrays[0].dtype

  n = np.prod([x.size for x in arrays])
  if out is None:
    out = np.zeros([n, len(arrays)], dtype=dtype)

  m = n / arrays[0].size
  out[:,0] = np.repeat(arrays[0], m)
  if arrays[1:]:
    cartesian(arrays[1:], out=out[0:m,1:])
    for j in xrange(1, arrays[0].size):
      out[j*m:(j+1)*m,1:] = out[0:m,1:]
  return out
print cartesian([[1,2],[3,4]])

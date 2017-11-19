


"""
k = 10
myColumn = matrix[:,0]
myColumn = np.array(myColumn)
myColumn = myColumn.astype(np.float)
max = np.amax(myColumn)
min = np.amin(myColumn)
size = (max - min)/k
bins = []
temp = min
for i in range(k):
    bins.append(temp)
    temp = temp + size
bins = np.array(bins)
binning = np.digitize(myColumn, bins)
print(binning[291])
"""
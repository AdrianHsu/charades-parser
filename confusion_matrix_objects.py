import csv
import numpy
import matplotlib.pyplot as plt

def find_key_idx( dict, term ):
  
  keys = list(dict.keys())
  for i in range( len(keys) ):
    if keys[i] == term:
      return i
  return -1


objects = {}

reader = csv.DictReader( open( "Charades_vu17_train.csv") )
for row in reader:
  obj = row["objects"]
  if obj == "":
    continue

  objs = obj.split(';')
  for o in objs:
    if o in objects.keys():
      objects[o] += 1
    else:
      objects[o] = 1

keys = list(objects.keys())
matrix = numpy.zeros( (len(keys), 157) )

reader = csv.DictReader( open( "Charades_vu17_train.csv") )
for row in reader:
  obj = row["objects"]
  if obj == "":
    continue
  act = row["actions"]
  if act == "":
    continue

  objs = obj.split(';')
  acts = act.split(';')

  for o in objs:
    for a in acts:
      tks = a.split()[0]
      lbl = int( tks[1:] )
      oid = find_key_idx( objects, o )

      matrix[ oid, lbl ] += 1

counts = numpy.load("class_counts.npy")
n1 = matrix / counts
n2 = n1[:]

for i in range( len(keys) ):
  key = keys[i]
  cnt = objects[key]
  n2[i,:] = n1[i,:]/cnt

res = numpy.sum( n2, axis=0 )

idx = numpy.argsort( res )[::-1]
print(res[idx[0:5]])
print(idx)

plt.imshow( n1, interpolation = 'None' )
plt.show()

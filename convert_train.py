import csv

valid = ["c060", "c103", "c083", "c088", "c041", "c044", "c046", "c049", "c128", "c129"]

reader= csv.DictReader( open( "train.csv" ) )
outputfile = open( "picked_train.txt", 'w' )
vfps = [l.strip() for l in open('video_fps.txt')]
vfps = { l.split()[0] : float( l.split()[1] ) for l in vfps }

for row in reader:
  vid = row['id']
  acts = row['actions']
  if acts == "":
    continue
  
  acts = acts.split(';')
  for a in acts:
    line = ""
    tokens = a.split()
    if tokens[0] not in valid:
      continue

    ts = float( tokens[1] )
    te = float( tokens[2] )
    ts = int( round( ts*vfps[vid] ) )
    te = int( round( te*vfps[vid] ) )
    line = line + tokens[0] + " " + str(ts) + " " + str(te)# + ";"
    outputfile.write( vid + "|" + line + "\n" )

  if line == "":
    continue
    
outputfile.close()

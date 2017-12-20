import csv

#valid = ["c059", "c011", "c151", "c156", "c149"]

reader= csv.DictReader( open( "test.csv" ) )
outputfile = open( "picked_test.txt", 'w' )
vfps = [l.strip() for l in open('video_fps.txt')]
vfps = { l.split()[0] : float( l.split()[1] ) for l in vfps }

for row in reader:
  vid = row['id']
  acts = row['actions']
  if acts == "":
    continue
  
  line = ""
  acts = acts.split(';')
  for a in acts:
    tokens = a.split()
#    if tokens[0] not in valid:
#      continue

    ts = float( tokens[1] )
    te = float( tokens[2] )
    ts = int( 0 )# round( ts*vfps[vid] ) )
    te = int( 0 )# round( te*vfps[vid] ) )
    line = line + tokens[0] + " " + str(ts) + " " + str(te) + ";"
    outputfile.write( vid + "|" + line + "\n" )

  if line == "":
    continue
    
outputfile.close()

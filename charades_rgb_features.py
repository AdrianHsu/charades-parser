import random
import h5py
import numpy
import tensorflow as tf

import util

SAMPLE_VIDEO_TRAIN = 10
SAMPLE_VIDEO_TEST  = 25           # Sample this many frames per video
BATCH = 40   # Process this many videos concurrently
LR = 1e-3
C = 157

PRINT_EVERY = 25

def compute_mAP( data, labels, num_classes=4 ):

  mAP = 0.
  for i in range( num_classes ):
    # Sort the activations for class i by magnitude
    dd = data[ :, i ]
    ll = labels[:,i]
    idx = numpy.argsort( -dd )

    # True positives, False positives
    tp = (ll == 1).astype( numpy.int32 )
    fp = (ll == 0).astype( numpy.int32 )

    # Number of instances with label i
    num= numpy.sum( ll )

    # In case a class has been completely filtered by preprocessing:
    # For example c136: No frames before action that is to be predicted
    if num == 0:
      continue

    # Reorder according to the sorting
    tp = tp[idx]
    fp = fp[idx]

    tp = numpy.cumsum( tp ).astype( numpy.float32 )
    fp = numpy.cumsum( fp ).astype( numpy.float32 )

    prec = tp / (tp + fp)

    ap = 0.
    tmp = ll[idx]
    for j in range( data.shape[0] ):

      ap += tmp[j] * prec[j]
    ap /= num

    mAP += ap

  return mAP / num_classes

def sample_points( ts, te, num ):
  assert( ts <= te )
  diff = te-ts
  pos = []
  for i in range( num ):    
    pos.append( ts + int(round(i*(diff/float(num)))) )
  return pos


# Convert to Charades evaluation script format
def save_to_eval( vidids, data, outputname = "output.txt" ):
  if len(vidids) != len(data):
    print("Parameters do not match")
    return

  outputfile = open( outputname, 'w' )

  for i in range( len(vidids) ):
    vid = vidids[i]
    vec = data[i].tolist()
    vec = [str(v) for v in vec]

    out = " "
    res = out.join( vec )
    outputfile.write( vid + " " + res + "\n\n" )

  outputfile.close()


# label_dict = { 'c008' : 0, 'c009' : 1, 'c081' : 2, 'c112' : 3 }   # For subset
label_dict = { 'c'+str(i).zfill(3) : i for i in range( 157 )}     # For full Dataset

# Unfortunately, the features were sampled at 24 fps --> We need to recover the correct frame number
fps_dict = { l.strip().split(' ')[0] : float( l.strip().split(' ')[1] ) for l in open( "video_fps.txt" ) }


PREFIX = "/tmp3/agethen/Predictive/CharadesRGBFeat/"

def load_data( files, sample_size = 1, is_test = False ):

  h5file = h5py.File( PREFIX + files[0].split('|')[0] + '.h5' )
  h5data = h5file['data'][:]
  dim = h5data.shape[1]
  h5file.close()

  data = numpy.zeros( (len(files) * sample_size, dim) )
  labels = numpy.zeros( (len(files) * sample_size, C) )
  for i,f in enumerate(files):

    # Annotation: VID|CID1 TS1 TE1;CID2 TS2 TE2
    # During training, we guarantee only one annotation per entry (to avoid multilabel class. problem)!
    ff = f.split('|')

    acts  = ff[1].split(';')
    if is_test == False:
      assert( len(acts) == 1 )

    ts    = []
    te    = []
    cids  = []
    for a in acts:
      tokens = a.split(' ')
      cids.append( tokens[0] )
      ts.append( int(tokens[1]) )
      te.append( int(tokens[2]) )

    # Get H5 data
    h5file = h5py.File( PREFIX + ff[0] + '.h5' )
    h5data = h5file['data'][:]
    h5file.close()    

    # Sampling evenly spaced
    if is_test == False:
      pos = sample_points( ts[0], te[0], sample_size )
    else:
      pos = sample_points( 0, h5data.shape[0], sample_size )

    pos = numpy.array( pos )
    pos = pos * (24.0 / fps_dict[ ff[0] ])
    pos = pos // 4                          # Every 4-th frame was sampled
    pos = pos.astype( numpy.int32 )

    for j,t in enumerate( pos ):
      t = min( t, h5data.shape[0]-1 )
      data[i*sample_size + j] = h5data[t]

      for c in cids:
        labels[i*sample_size + j, label_dict[ c ]] = 1.

  return data, labels

X = tf.placeholder( "float", [None, 4096] )
Y = tf.placeholder( "int32", [None, C] )

fc1 = util.fc( X, C, "fc1" )
pre = tf.nn.softmax_cross_entropy_with_logits( logits = fc1, labels = Y )
loss= tf.reduce_mean( pre )

optimizer  = tf.train.AdamOptimizer( learning_rate = LR, epsilon=1e-8 )
gradvars   = optimizer.compute_gradients( loss )

capped     = [(tf.clip_by_value( grad, -5, 5 ), var) for grad, var in gradvars]
train_op   = optimizer.apply_gradients( capped )


conf = tf.ConfigProto(
      gpu_options = tf.GPUOptions( allow_growth = True ),
      device_count = { 'GPU': 1 }
    )


train_files = [l.strip() for l in open( "picked_train.txt")]
test_files = [l.strip() for l in open( "picked_test.txt")]

num_train = len(train_files)
num_test  = len(test_files)
test_vids = [l.split('|')[0] for l in test_files]

with tf.Session( config = conf ) as sess:
  tf.global_variables_initializer().run()

  for e in range( 100 ):
    print("\nEpoch", str(e).zfill(3))

    # Train Phase.
    random.shuffle( train_files )

    cnt_iteration = 0
    for start, end in list( zip( range( 0, num_train, BATCH ), range( BATCH, num_train+1, BATCH ) )):
      data, labels = load_data( train_files[ start : end ], sample_size = SAMPLE_VIDEO_TRAIN )

      tf_op, tf_loss = sess.run( [train_op, loss], feed_dict = { X : data, Y : labels } )

      if cnt_iteration % PRINT_EVERY == 0:
        print("Train Phase\t", "Iteration", str(start).zfill(5), "\tLoss", tf_loss)
      cnt_iteration += 1


    # Test Phase. Make sure not to shuffle, otherwise `test_vids` won't match.
    results  = numpy.zeros( (num_test, C) )
    gt       = numpy.zeros( (num_test, C) )

    test_range = list(zip( range( 0, num_test, BATCH ), range( BATCH, num_test+1, BATCH ) ))

    if num_test % BATCH != 0:
      test_range.append( (num_test-(num_test%BATCH), num_test) )
    

    for start, end in test_range:
      data, labels = load_data( test_files[ start : end ], sample_size = SAMPLE_VIDEO_TEST, is_test = True )

      tf_res = sess.run( fc1, feed_dict = { X : data } )

      # During test phase, we pool the SAMPLE_VIDEO results of each video (averaging)
      tf_res = numpy.reshape( tf_res, [-1, SAMPLE_VIDEO_TEST, C] )
      tf_res = numpy.average( tf_res, axis=1 )
      labels = numpy.reshape( labels, [-1, SAMPLE_VIDEO_TEST, C] )[:,0,:]

      results[ start : end ]  = tf_res
      gt[ start : end ]       = labels

    mAP = compute_mAP( results, gt, num_classes = C )

    save_to_eval( test_vids, results )

    print("Test Phase -- Mean AP: mAP =", mAP)

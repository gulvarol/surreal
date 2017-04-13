import numpy as np
import pickle


# PARAMS
stepsize = 4 # subsamping 120Hz to 30Hz
smpl_data = np.load("../smpl_data/smpl_data.npz")
split_dir = "splits"

split_test_filename = "001_test.txt"
split_train_filename = "001_train.txt"


print "Parsing cmu keys.."
cmu_keys = []
for seq in smpl_data.files:
	if seq.startswith('pose_'):
		cmu_keys.append(seq.replace('pose_', ''))

#name = sorted(cmu_keys)[idx % len(cmu_keys)]

cmu_keys.sort()

print "Loading cmu data.."
cmu_parms = {}
for seq in smpl_data.files:
	if seq.startswith('pose_'):
		cmu_parms[seq.replace('pose_', '')] = {'poses':smpl_data[seq],
		                                      'trans':smpl_data[seq.replace('pose_','trans_')]}


print "Parsing splits.."
split_test = []
split_train = []
# parse split
with open(split_dir + '/' + split_test_filename) as f:
	split_test = f.read().splitlines()
with open(split_dir + '/' + split_train_filename) as f:
	split_train = f.read().splitlines()


print "Create seq_info for every sequence (name, nb_frames, use_split).."
seq_info = []
for seq in cmu_keys:
	nb_frames = len(cmu_parms[seq]['poses'])/float(stepsize)
	
	# -- possible seq names:
	# ung_105_22  -> train/test
	# 01_02  -> train/test
	# h36m_S11_Directions 1  -> all
	use_split = ''
	
	name = seq.split('_')
	firstchunk = name[0]
	
	if firstchunk == "h36m":
		use_split = 'all'
	else:
		if firstchunk == "ung":
			firstchunk = name[1]
		
		# cast as integer to get subject ID
		subject_ID = "%02d" % int(firstchunk)
		if subject_ID in split_test:
			use_split = 'test'
		elif subject_ID in split_train:
			use_split = 'train'
		else:
			assert False
	
	
	seq_info.append({"name": seq, "nb_frames": nb_frames, "use_split": use_split})


print "Save as pickle and txt.."
# Save as pickle
with open("../pkl/idx_info.pickle", "wb") as f:
	pickle.dump(seq_info, f)

with open("idx_info.txt", "w") as f:
	f.write("# idx  nb_frames use_split name\n")
	for i, elem in enumerate(seq_info):
		f.write("%d %d %s %s\n" % (i, elem['nb_frames'], elem['use_split'], elem['name'] ))


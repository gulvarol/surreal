import pickle
import numpy as np

run_stride = [50, 30, 70]

stepsize = 4
clipsize = 100


fout = open("job_list.txt", "w")

all_idx_info = pickle.load(open("../pkl/idx_info.pickle", 'rb'))
nb_idx = len(all_idx_info)

for it_runpass, stride in enumerate(run_stride):
	
	for idx, idx_info in enumerate(all_idx_info):
		nb_ishape = max(1, int(np.ceil((idx_info['nb_frames'] - (clipsize - stride))/stride)))
		
		for it_ishape in range(nb_ishape):
			fout.write("--idx %d --ishape %d --stride %d\n" % (idx + it_runpass * nb_idx, it_ishape, stride))
		#(nb)divmod(int(idx_info['nb_frames']), clipsize)

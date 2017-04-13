import ast
import configparser
import pdb
import numpy as np
import random

def load_file(filename='config', section='SYNTH_DATA'):
	# returns dictionary with all params
	
	# Import configuration
	config = configparser.ConfigParser()
	res = config.read(filename)
	if len(res) == 0:
		print("ERROR: couldn't load 'config' file. To fix, copy 'config.sample' to 'config' (and do not version this file)")
		exit(1)
	
	params = {}
	options = config.options(section)
	for option in options:
		try:
			params[option] = ast.literal_eval(config.get(section, option))
			if params[option] == -1:
				print("skip: %s" % option)
		except:
			print(" CONFIG PARSING EXCEPTION on %s" % option)
			params[option] = None
			raise
			
	return params

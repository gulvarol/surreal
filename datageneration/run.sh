#!/bin/bash

JOB_PARAMS=${1:-'--idx 0 --ishape 0 --stride 50'} # defaults to [0, 0, 50]

# SET PATHS HERE
FFMPEG_PATH=/home/gvarol/tools/ffmpeg/ffmpeg_build_sequoia_h264
X264_PATH=/home/gvarol/tools/ffmpeg/x264_build/
PYTHON2_PATH=/home/gvarol/tools/anaconda/envs/surreal_env/ # PYTHON 2
BLENDER_PATH=/home/gvarol/tools/blender/blender-2.78a-linux-glibc211-x86_64
cd surreal/datageneration

# BUNLED PYTHON
BUNDLED_PYTHON=${BLENDER_PATH}/2.78/python
export PYTHONPATH=${BUNDLED_PYTHON}/lib/python3.4:${BUNDLED_PYTHON}/lib/python3.4/site-packages
export PYTHONPATH=${BUNDLED_PYTHON}:${PYTHONPATH}

# FFMPEG
export LD_LIBRARY_PATH=${FFMPEG_PATH}/lib:${X264_PATH}/lib:${LD_LIBRARY_PATH}
export PATH=${FFMPEG_PATH}/bin:${PATH}


### RUN PART 1  --- Uses python3 because of Blender
$BLENDER_PATH/blender -b -t 1 -P main_part1.py -- ${JOB_PARAMS}

### RUN PART 2  --- Uses python2 because of OpenEXR
PYTHONPATH="" ${PYTHON2_PATH}/bin/python2.7 main_part2.py ${JOB_PARAMS}

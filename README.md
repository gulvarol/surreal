# Learning from Synthetic Humans (SURREAL)

This is the code for the following paper:

Gül Varol, Javier Romero, Xavier Martin, Naureen Mahmood, Michael J. Black, Ivan Laptev and Cordelia Schmid, [Learning from Synthetic Humans](https://arxiv.org/abs/1701.01370), CVPR 2017.

Check the [project page](http://www.di.ens.fr/willow/research/surreal/) for more materials.

Contact: [Gül Varol](http://www.di.ens.fr/~varol/).

## 1. Download SURREAL dataset
In order to download SURREAL dataset, you need to accept the license terms. The links to license terms and download procedure are available here:

https://www.di.ens.fr/willow/research/surreal/data/

Once you receive the credentials to download the dataset, you will have a personal username and password. Use these with the `download/download_surreal.sh` script as follows:

``` shell
./download_surreal.sh /path/to/dataset yourusername yourpassword
```

The structure of the folders is as follows:

``` shell
SURREAL/data/
------------- h36m/ # using MoCap from Human3.6M dataset
------------- cmu/  # using MoCap from CMU dataset
-------------------- train/
-------------------- test/
----------------------------  run0/ #50% overlap
----------------------------  run1/ #30% overlap
----------------------------  run2/ #70% overlap
------------------------------------  <sequenceName>/ #e.g. 01_01 for cmu, h36m_S1_Directions for h36m
--------------------------------------------------  <sequenceName>_c%04d.mp4       # RGB - 240x320 resolution video
--------------------------------------------------  <sequenceName>_c%04d_depth.mat # Depth
#     depth_1, depth_2, ... depth_T [240x320 single] - in meters
--------------------------------------------------  <sequenceName>_c%04d_segm.mat  # Segmentation
#     segm_1,   segm_2, ...  segm_T [240x320 uint8]  - 0 for background and 1..24 for SMPL body parts
--------------------------------------------------  <sequenceName>_c%04d_info.mat  # Remaining annotation
#     bg           [1xT cell]      - names of background image files
#     camDist      [1 single]      - camera distance
#     camLoc       [3x1 single]    - camera location
#     clipNo       [1 double]      - clip number of the full sequence (corresponds to the c%04d part of the file)
#     cloth        [1xT cell]      - names of texture image files
#     gender       [Tx1 uint8]     - gender (0: 'female', 1: 'male')
#     joints2D     [2x24xT single] - 2D coordinates of 24 SMPL body joints on the image pixels
#     joints3D     [3x24xT single] - 3D coordinates of 24 SMPL body joints in real world meters
#     light        [9x100 single]  - spherical harmonics lighting coefficients
#     pose         [72xT single]   - SMPL parameters
#     sequence     [char]          - <sequenceName>_c%04d
#     shape        [10xT single]   - body shape parameters
#     source       [char]          - 'cmu' or 'h36m'
#     stride       [1 uint8]       - percent overlap between clips, 30 or 50 or 70
#     zrot         [Tx1 single]    - rotation in Z

# *** T is the number of frames, mostly 100.

```

## 2. Create your own synthetic data
### Preparation
#### SMPL data

a) You need to download SMPL for MAYA from http://smpl.is.tue.mpg.de in order to run the synthetic data generation code. Once you agree on SMPL license terms and have access to downloads, you will have the following two files:

```
basicModel_f_lbs_10_207_0_v1.0.2.fbx
basicModel_m_lbs_10_207_0_v1.0.2.fbx
```

Place these two files under `datageneration/smpl_data` folder.

b) With the same credentials as with the SURREAL dataset, you can download the remaining necessary SMPL data and place it in `datageneration/smpl_data`.

``` shell
./download_smpl_data.sh /path/to/smpl_data yourusername yourpassword
```

 * `textures/` - folder containing clothing images
 * `smpl_data.npz` - [MoSH](http://mosh.is.tue.mpg.de/)'ed CMU and Human3.6M MoCap data (4GB)
 * `(fe)male_beta_stds.npy`

#### Background images

We only provide names of the background images we used. They are downloaded from [LSUN dataset](http://lsun.cs.princeton.edu/2016/index.html) using [this code](https://github.com/fyu/lsun). You can download images from this dataset or use any other images.

#### Blender
You need to download [Blender](http://download.blender.org/release/) and install scipy package to run the first part of the code. The provided code was tested with [Blender2.78](http://download.blender.org/release/Blender2.78/blender-2.78a-linux-glibc211-x86_64.tar.bz2), which is shipped with its own python executable as well as distutils package. Therefore, it is sufficient to do the following:

``` shell
# Install pip
/blenderpath/2.78/python/bin/python3.5m get-pip.py
# Install scipy
/blenderpath/2.78/python/bin/python3.5m pip install scipy
```

`get-pip.py` is downloaded from [pip](https://pip.pypa.io/en/stable/installing/). Replace the `blenderpath` with your own and set `BLENDER_PATH`.

Otherwise, you might need to point to your system installation of python, but be prepared for unexpected surprises due to version mismatches. There may not be support about questions regarding this installation.

#### FFMPEG
If you want to save the rendered images as videos, you will need [ffmpeg](https://ffmpeg.org/) library. Build it and set the `FFMPEG_PATH` to the directory that contains `lib/` and `bin/` folders. Additionally, if you want to use H.264 codec as it is done in the current version of the code, you need to have the [x264](http://www.videolan.org/developers/x264.html) libraries compiled. In that case, set `X264_PATH` to your build. If you use another codec, you don't need `X264_PATH` variable and you can remove `-c:v h264` from `main_part1.py`.

This is how the ffmpeg was built:

``` shell
# x264
./configure  --prefix=/home/gvarol/tools/ffmpeg/x264_build --enable-static --enable-shared --disable-asm
make 
make install

# ffmpeg
./configure --prefix=/home/gvarol/tools/ffmpeg/ffmpeg_build_sequoia_h264 --enable-avresample --enable-pic --disable-doc --disable-static --enable-shared --enable-gpl --enable-nonfree --enable-postproc --enable-x11grab --disable-yasm --enable-libx264 --extra-ldflags="-I/home/gvarol/tools/ffmpeg/x264_build/include -L/home/gvarol/tools/ffmpeg/x264_build/lib" --extra-cflags="-I/home/gvarol/tools/ffmpeg/x264_build/include"
make
make install
```

#### OpenEXR
The file type for some of the temporary outputs from Blender will be EXR images. In order to read these images, the code uses [OpenEXR bindings for Python](http://www.excamera.com/sphinx/articles-openexr.html). These bindings are available for python 2, the second part of the code (`main_part2.py`) needs this library.

### Running the code
Copy the `config.copy` into `config` and edit the `bg_path`, `tmp_path`, `output_path` and `openexr_py2_path` with your own paths.

* `bg_path` contains background images and two files `train_img.txt` and `test_img.txt`. The ones used for SURREAL dataset can be found in `datageneration/misc/LSUN`. Note that the folder structure is flattened for each room type.

* `tmp_path` stores temporary outputs and is deleted afterwards. You can use this for debugging.

* `output_path` is the directory where we store all the final outputs of the rendering.

* `openexr_py2_path` is the path to libraries for [OpenEXR bindings for Python](http://www.excamera.com/sphinx/articles-openexr.html).

`run.sh` script is ran for each clip. You need to set `FFMPEG_PATH`, `X264_PATH` (optional), `PYTHON2_PATH`, and `BLENDER_PATH` variables. `-t 1` option can be removed to run on multi cores, it runs faster.

 ``` shell
# When you are ready, type:
./run.sh
```

## 3. Training models
TO-DO

## Citation
If you use this code, please cite the following:
> @article{varol17a,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TITLE = {Learning from Synthetic Humans},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AUTHOR = {Varol, G{\"u}l and Romero, Javier and Martin, Xavier and Mahmood, Naureen and Black, Michael J. and Laptev, Ivan and Schmid, Cordelia},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;JOURNAL =  {CVPR},  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;YEAR = {2017}  
}

## Acknowledgements
The data generation code is built by [Javier Romero](https://github.com/libicocco/), [Gul Varol](https://github.com/gulvarol) and [Xavier Martin](https://github.com/xavier-martin).


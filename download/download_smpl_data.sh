outdir=${1:-'/tmp'}
username=${2:-'username'}
password=${3:-'password'}

# This script can be used to download necessary data to run data generation code.
# It includes:
#     * textures/ - folder containing clothing images
#     * smpl_data.npz - MoSh'ed CMU MoCap data (2.5GB)
#     * (fe)male_beta_stds.npy
#
# Run `chmod u+x download_smpl_data.sh` and pass the path of the output directory as follows:
# `./download_smpl_data.sh /path/to/datageneration yourusername yourpassword` 
# Replace the path with your download folder, potentially '../datageneration'.
# Replace username and password with the credentials you received by e-mail upon accepting license terms.
# Place smpl_data folder under the data generation code folder.
# You can remove -q option to debug.

wget --user=${username} --password=${password} -m -q -i files/files_smpl_data.txt --no-host-directories -P ${outdir}

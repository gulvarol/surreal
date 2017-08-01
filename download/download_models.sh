outdir=${1:-'/tmp'}
username=${2:-'username'}
password=${3:-'password'}

# This script can be used to download pre-trained models on SURREAL.
#
# Run `chmod u+x download_models.sh` and pass the path of the output directory as follows:
# `./download_models.sh /path/to/models yourusername yourpassword` 
# Replace the path with your download folder.
# Replace username and password with the credentials you received by e-mail upon accepting license terms.
# You can remove -q option to debug.

wget --user=${username} --password=${password} -m -q -i files/files_models.txt --no-host-directories -P ${outdir}

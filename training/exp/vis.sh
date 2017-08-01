testNo=$1
modelNo=$2
dataset=${3:-'cmu'}
testDir=${4:-'val'}
evaluate=${5:-'-'}

fixedStr="qlua main.lua \
-training pretrained \
-epochSize 0 \
-batchSize 1 \
-nDonkeys 0 \
-verbose \
-show \
-datasetname ${dataset} \
-testDir ${testDir} \
-dirName vis"


if [ $evaluate = "eval" ]
then
	fixedStr="$fixedStr -evaluate "
fi

if [ $testNo -eq 1 ]
then
	cmd="$fixedStr \
	-supervision segm \
	-retrain ~/cnn_saves/cmu/segmscratch/model_${modelNo}.t7"
fi

if [ $testNo -eq 2 ]
then
	cmd="$fixedStr \
	-supervision depth \
	-retrain ~/cnn_saves/cmu/depthscratch/model_${modelNo}.t7"
fi

printf "\n\n\nRunning...\n\n"
printf "<=================================================>\n"
printf "<=================================================>\n"
printf "\n\n$cmd\n\n"
printf "<=================================================>\n"
printf "<=================================================>\n\n"
eval $cmd

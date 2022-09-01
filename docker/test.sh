#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="10g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

docker volume create petctsegmentationcontainer-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --gpus="all"  \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/input:/input/ \
        -v petctsegmentationcontainer-output-$VOLUME_SUFFIX:/output/ \
        petctsegmentationcontainer

echo "Evaluation done, checking results"
docker build -f Dockerfile.eval -t unet_eval .

docker run --rm \
        -v petctsegmentationcontainer-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test/expected_output/:/expected_output/ \
        unet_eval python3 -c """
import SimpleITK as sitk
import os

print('Start')
file = os.listdir('/output/images/automated-petct-lesion-segmentation')[0]
print(file)
output = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join('/output/images/automated-petct-lesion-segmentation/', file)))
expected_output = sitk.GetArrayFromImage(sitk.ReadImage('/expected_output/PRED.nii.gz'))
mse = sum(sum(sum((output - expected_output) ** 2)))
if mse <= 10:
    print('Test passed!')
else:
    print(f'Test failed! MSE={mse}')
dice = 2*(output*expected_output).sum()/(output.sum()+expected_output.sum())
if dice >= 0.5:
    print('Test passed!')
else:
    print(f'Test failed! dice={dice}')
"""
docker volume rm petctsegmentationcontainer-output-$VOLUME_SUFFIX

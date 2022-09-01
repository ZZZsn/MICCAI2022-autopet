#!/usr/bin/env bash

./build.sh

docker save petctsegmentationcontainer | gzip -c > PetCtSegmentationContainer.tar.gz

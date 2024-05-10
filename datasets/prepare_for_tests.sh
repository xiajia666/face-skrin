#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates.

# Download the mini dataset (balloon val2017_100, with only 100 images)
# to be used in unittests & integration tests.

cd "${0%/*}"

BASE=https://dl.fbaipublicfiles.com/detectron2
ROOT=${DETECTRON2_DATASETS:-./}
ROOT=${ROOT/#\~/$HOME}   # expand ~ to HOME
mkdir -p $ROOT/balloon/annotations

for anno in instances_val2017_100 \
  person_keypoints_val2017_100 ; do

  dest=$ROOT/balloon/annotations/$anno.json
  [[ -s $dest ]] && {
    echo "$dest exists. Skipping ..."
  } || {
    wget $BASE/annotations/balloon/$anno.json -O $dest
  }
done

dest=$ROOT/balloon/val2017_100.tgz
[[ -d $ROOT/balloon/val2017 ]] && {
  echo "$ROOT/coco/val2017 exists. Skipping ..."
} || {
  wget $BASE/annotations/balloon/val2017_100.tgz -O $dest
  tar xzf $dest -C $ROOT/balloon/ && rm -f $dest
}

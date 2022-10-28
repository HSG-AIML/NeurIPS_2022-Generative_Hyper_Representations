#!/bin/bash
clear
echo Downloading Hyper-Representations from https://zenodo.org/record/7261342/files/hyper_reps.zip?download=1

curl -# -o "hyper_reps.zip" "https://zenodo.org/record/7261342/files/hyper_reps.zip?download=1"

echo Unzipping hyper_reps

unzip hyper_reps.zip

echo Downloading pre-packaged zoos from https://zenodo.org/record/7261342/files/zoos.zip?download=1

curl -# -o "zoos.zip" "https://zenodo.org/record/7261342/files/zoos.zip?download=1"

echo Unzipping zoos

unzip zoos.zip

echo done

#!/bin/bash

# directory=$1

# for file in "$directory"/*.mp4; do
#     output="${file//.mp4/.pose}"
#     echo $output
#     sbatch job.sh video_to_pose -i $file --format mediapipe -o $output
# done

# for file in "$directory"/*.mp4; do
#     name=$(basename "$file" ".mp4")
#     # echo $name
#     if grep -Fxq $name /shares/volk.cl.uzh/zifjia/subtitle_align/data/bobsl_align.txt
#     then
#         output="${file//.mp4/.pose}"
#         echo $output
#         sbatch job.sh video_to_pose -i $file --format mediapipe -o $output
#     fi
# done

directory_in=$1
directory_out=$2

for file_in in $(find "$directory_in" -name '*.mpg' -or -name '*.mp4'); do
    file_out="${file_in//.mpg/.pose}"
    file_out=${file_out##*/}
    file_out="$directory_out$file_out"
    echo $file_in
    echo $file_out
    video_to_pose -i $file_in --format mediapipe -o $file_out
done

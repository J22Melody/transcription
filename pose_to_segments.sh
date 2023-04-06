directory=$1

for file in "$directory"/*.pose; do
    output="${file//.pose/.seg.npy}"
    echo $output
    sbatch job.sh pose_to_segments -i $file -o $output -f probs
done

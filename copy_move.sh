#!/bin/bash

copy_and_rename() {
    local src_dir=$1
    local prefix=$2
    local subdirs=("adenocarcinoma" "large.cell.carcinoma" "normal" "squamous.cell.carcinoma")
    # echo "$src_dir"
    for sub_dir in "$src_dir"/*/; do
        # echo "$sub_dir"
        for file in "$sub_dir"/*; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                #put it in the correct target subdir
                for label in "${subdirs[@]}"; do
                    if [[ "$sub_dir" == *"$label"* ]]; then
                        cp "$file" "data/all_data/${label}/${prefix}_$filename"
                    fi
                done
            fi
        done 
    done
}

#add the second positional arg!!!!
copy_and_rename "data/train" "train"
copy_and_rename "data/test"  "test"
copy_and_rename "data/valid" "valid"
echo "All done!"
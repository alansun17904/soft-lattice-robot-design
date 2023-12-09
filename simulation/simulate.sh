#!/bin/bash
total=$(ls robots/*.json | wc -l)

for fname in robots/*.json; do
    echo $fname
    python3 mass_spring_reverse.py "$fname" "train"
done # | tqdm --total $total >> /dev/null

#tqdm --total $total >> /dev/null
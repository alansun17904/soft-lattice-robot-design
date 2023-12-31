#!/bin/bash
total=$(ls robots/*.json | wc -l)

for fname in robots/*.json; do
    echo $fname
    python3 mass_spring.py "$fname" "plot" "losses-test.json" "flat.png"
done # | tqdm --total $total >> /dev/null

#tqdm --total $total >> /dev/null
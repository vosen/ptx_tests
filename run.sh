#!/bin/bash

for i in $(seq 0 $(($2 - 1))); do
    cargo run -r -- $1 --shard-index $i --shard-count $2 &> output_$i.log &
done
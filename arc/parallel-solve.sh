#!/bin/bash
for n in $(seq $1); do
  LOG="logs/out-${n}.log"
  python solve_labyrinth.py --alg=her-sac --env=bullet_hard --mode=train --num_steps=500000 --seed=$n --random_starts &>"${LOG}" &
done

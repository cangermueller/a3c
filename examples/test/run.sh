#!/usr/bin/env bash

set -e
shopt -s extglob

check=1
function run {
  cmd=$@
  echo
  echo "#################################"
  echo $cmd
  echo "#################################"
  eval $cmd
  if [ $check -ne 0 -a $? -ne 0 ]; then
    1>&2 echo "Command failed!"
    exit 1
  fi
}

# FrozenLake-v0
# ==============
nb_hidden=64
nb_rnn_unit=$nb_hidden
out_dir="./results"
cmd="python -u $(which run_a3c.py)
  --env FrozenLake-v0
  --save_freq 1000
  --out_checkpoint $out_dir/checkpoints/00
  --tensorboard $out_dir/tensorboard
  --monitor $out_dir/monitor

  --nb_train 100
  --nb_play 0

  --nb_agent 1
  --learning_rate 0.001
  --rollout_len 5
  --gamma 0.99
  --lambd 0.99
  --entropy_weight 0.01
  --huber_loss

  --nb_hidden $nb_hidden
  --nb_rnn_unit $nb_rnn_unit
  "
run $cmd

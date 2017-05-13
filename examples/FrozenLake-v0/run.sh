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
# https://gym.openai.com/evaluations/eval_C3Zl2BQSCnbunQRA8ibg

nb_hidden=64
nb_rnn_unit=$nb_hidden
out_dir="./results"
cmd="python -u $(which run_a3c.py)
  --env FrozenLake-v0
  --out_checkpoint $out_dir/checkpoints/00
  --monitor $out_dir/monitor
  --save_freq 1000

  --nb_train 5000
  --nb_play 2

  --nb_agent 4
  --learning_rate 0.001
  --rollout_len 5
  --discount 0.99
  --entropy_weight 0.01

  --nb_hidden $nb_hidden
  --nb_rnn_unit $nb_rnn_unit
  "
run $cmd

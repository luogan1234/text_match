for seed in {0..4}; do
  for lr0 in 3e-4 2e-4 1e-4; do
    for lr in 7e-5 6e-5 5e-5; do
      cmd="python main.py -train train -test testA -text_encoder bert -lr0 $lr0 -lr $lr -seed $seed"
      echo $cmd & $cmd
    done
  done
done

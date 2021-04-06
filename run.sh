for seed in {0..5}; do
  for lr in 4e-5 5e-5 6e-5; do
    for lr0 in 3e-4; do
      for mask_w in 1.0 1.5 2.0; do
        cmd="python main.py -train train -test testA -lr0 $lr0 -lr $lr -mask_w $mask_w -seed $seed"
        echo $cmd & $cmd
      done
    done
  done
done

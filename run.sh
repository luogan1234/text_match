python init.py
python feature.py
for seed in {0..5}; do
  cmd="python main.py -train train -test testA -lr0 3e-4 -lr 5e-5 -mask_w 1.5 -seed $seed"
  echo $cmd & $cmd
done
python predict.py
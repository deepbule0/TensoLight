{
    objects=(realdog1 realdog2 realbear realchair)
    for object in ${objects[@]}; do
        export PYTHONPATH=. && python train.py  --config ./configs/real/${object}.txt
    done
    exit
}
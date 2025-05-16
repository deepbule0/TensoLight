{
    objects=(head hotdog teapot)
    for object in ${objects[@]}; do
        export PYTHONPATH=. && python train.py  --config ./configs/syn/${object}.txt
    done
    exit
}
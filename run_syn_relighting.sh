{
    objects=(head teapot hotdog)
    for object in ${objects[@]}; do
        export PYTHONPATH=. && python relight.py  --config ./configs/relighting/${object}.txt
    done
    exit
}
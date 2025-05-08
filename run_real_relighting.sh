{
    objects=(realdog1 realdog2 realbear realchair)
    for object in ${objects[@]}; do
        export PYTHONPATH=. && python relight.py  --config ./configs/relighting/${object}.txt
    done
    exit
}
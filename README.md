# TensoLight: Inverse Rendering with Tensorial Radiance Fields based Environment Lighting Model


##  Dependencies

Install environment as follows:

```
conda create -n TensoLight python=3.8
conda activate TensoLight
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard loguru plyfile mitsuba tensorboard jaxtyping matplotlib
```

## Dataset

 Please download the dataset from the following links and unzip to the project directory. 
 
https://github.com/deepbule0/TensoLight/releases/download/v1.0/data.zip


The data.zip file contains synthetic and real-world datasets for evaluation purposes. We converted the HDR image data from the NeRF Emitter into the sRGB format through tone mapping and generated foreground masks for the real-world datasets using SAM2 (Segment Anything Model v2).


## Training and Evaluation

To perform training and evaluation on both synthetic and real-world datasets, simply run:

```bash
bash run_all.sh
```


For relighting results on the real-world datasets, use:

```bash
bash run_real_relight.sh
```

## Acknowledgement

The code was built on [TensoIR](https://github.com/Haian-Jin/TensoIR) and [Nerf Emitter](https://github.com/gerwang/nerf-emitter). Thanks for these great projects.


## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

![CC BY 4.0 License](https://i.creativecommons.org/l/by/4.0/88x31.png)

<!-- [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] -->
# TensoLight: Inverse Rendering with Tensorial Radiance Fields based Environment Lighting Representation


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

Simply run the following code for training and evaluation on the synthetic and real-world datasets.

```bash
bash run_all.sh
```


## Acknowledgement

The code was built on [TensoIR](https://github.com/Haian-Jin/TensoIR) and [Nerf Emitter](https://github.com/gerwang/nerf-emitter). Thanks for these great projects.

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

[![CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

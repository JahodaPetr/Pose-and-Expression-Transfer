# Pose and Expression Transfer between face portraits

Given a source and target face portrait, the designed network pro-
duces an output image where the pose and expression from the source face
image are transferred onto the target identity. The architecture consists of
two encoders and a mapping network that maps the two inputs into the latent
space of StyleGAN2, which generates a high-quality image.

![Pose and Expression Transfer](/doc/Diagram.png)


## Requirements
* 10 GB of free space
* High-end NVIDIA GPU with atleast 6 GB memory and support for CUDA 11
* NVIDIA drivers >= 450.80.02	

This work utilizes pytorch implementation of [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch) which requires a very specific environment.
It requires separate CUDA development toolkit for compiling custom ops on the fly using NVCC. We provide the required toolkit in the environment file, and therefore the environment installation may take some time.

Tested on Ubuntu 20.04 and Debian 10

## Installation

Clone this repository.
```
git clone https://github.com/JahodaPetr/Pose-and-Expression-Transfer
cd Pose-and-Expression-Transfer
```

Install dependencies
```
conda env create -f environment.yml
```

Activate envirnoment
```
conda activate exptransfer
```

Download required models :
  * [Pretrained network](https://drive.google.com/file/d/11q5_a0NAceAeQ7-WyH8hjIROcYwlTAax/view?usp=share_link)
  * [ReStyle](https://drive.google.com/file/d/1sw6I2lRIB0MpuJkpc8F5BJiSZrc0hjfE/view)
and save them to pretrained_models folder

## Inference

The `aligned` option is set when the images input images are already aligned to skip the alignment process
The `latent` option is set when the input identity image is already inverted to skip the inversion
The `coupled` option is set when the output image should contain the source, target, and generated image side by side


### Pose and expression transfer with latent code

```
python inference.py \
--output_path=experiment \
--checkpoint_path=pretrained_models/best_model.pt \
--source=aligned_images/3.png \
--latent=latents/1.pkl \
--aligned=True 
```

### Pose and expression transfer with identity image

```
python inference.py \
--output_path=experiment \
--checkpoint_path=pretrained_models/best_model.pt \
--source=aligned_images/3.png \
--target=aligned_images/1.png \
--aligned=True 
```

### Pose and expression transfer with random identity generated by StyleGAN


```
python inference.py \
--output_path=experiment \
--checkpoint_path=pretrained_models/best_model.pt \
--source=aligned_images/2.png \
--random_identity=True \
--aligned=True 
```

Additional notes: 
* when using custom images, set aligned to False
* Restyle network is necessary for using image target identity image or random identity generated via StyleGAN, but not for the first case, when latent code is already pre-calculated 

## Acknowledgements

Source code utilizes parts of and modifies the work of Richardson et al. [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel).
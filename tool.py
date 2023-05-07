import os
from argparse import Namespace

import time
from psp import MypSp, RestylepSp
import numpy as np
import torch
from PIL import Image
import sys
import pickle

sys.path.append(".")
sys.path.append("..")

from common import (
    tensor2im,
    get_transforms,
    align_face_npy_with_params,
    run_on_batch,
    get_average_image,
)


class Tool:
    def __init__(
        self,
        opts=None,
        result_path="./experiment/",
        checkpoint_path="./pretrained_models/best_model.pt",
    ):
        if opts is None:
            self.opts = Namespace()
            self.opts.input_nc = 3
            self.opts.device = "cuda:0"
            self.opts.test_batch_size = 1
            self.opts.checkpoint_path = checkpoint_path
            self.opts.result_path = result_path
            self.opts.output_size = 1024
            self.opts.resize_outputs = False
        else:
            self.opts = opts
            self.opts.result_path = result_path

        os.makedirs(result_path, exist_ok=True)
        self.net = MypSp(self.opts)

    def invert(self, target_path, aligned, target_image):
        """
        Inverts image - finds corresponding latent code to that image
        Uses ReStyle Encoder
        """
        ## Open and align image if necessary and if it's not a random idenitity
        if not target_image:
            if aligned:
                target_image = Image.open(target_path)
            else:
                print("Starting alignment of target image")
                tic = time.time()
                target_npy = align_face_npy_with_params(target_path)
                toc = time.time()
                print("Alignment time:", toc - tic, "s", sep=" ")
                target_image = Image.fromarray(target_npy)
                aligned_image_path = os.path.join(
                    "aligned_images", os.path.basename(target_path)
                )
                Image.fromarray(target_npy).save(aligned_image_path)

        os.makedirs("latents", exist_ok=True)

        ckpt = torch.load(
            "pretrained_models/restyle_psp_ffhq_encode.pt", map_location="cuda:0"
        )
        opts = ckpt["opts"]
        opts = Namespace(**opts)
        opts.checkpoint_path = "pretrained_models/restyle_psp_ffhq_encode.pt"
        opts.resize_outputs = False
        net = RestylepSp(opts)

        net.eval()
        net.cuda()

        transform = get_transforms()
        target_image = target_image.convert("RGB")
        target_image = transform(target_image)

        # get the image corresponding to the latent average
        avg_image = get_average_image(net, opts)
        with torch.no_grad():

            tic = time.time()
            print("Starting inverison")
            input_cuda = target_image.cuda().float().unsqueeze(0)
            result_batch, result_latents = run_on_batch(
                input_cuda, net, opts, avg_image
            )
            toc = time.time()
            print("Image inversion time:", toc - tic, "s", sep=" ")
        latent = result_latents[0][-1]
        latent_img = result_batch[0][-1]

        with open(
            os.path.join("latents", os.path.basename(target_path)[:-4] + ".pkl"),
            "wb",
        ) as f:
            pickle.dump(latent, f)

        tensor2im(latent_img).save(
            os.path.join("latents", os.path.basename(target_path))
        )
        return torch.from_numpy(latent)

    def run(self, source_path, target_path, latent_path, random_identity, aligned, coupled): 
        """
        Generates the output image from source and target image
        """
        if aligned:
            source_image = Image.open(source_path)
        else:
            os.makedirs("./aligned_images", exist_ok=True)
            print("Starting alignment of source image")
            tic = time.time()
            source_npy = align_face_npy_with_params(source_path)
            toc = time.time()
            print("Alignment time:", toc - tic, "s", sep=" ")
            aligned_image_path = os.path.join(
                "aligned_images", os.path.basename(source_path)
            )
            Image.fromarray(source_npy).save(aligned_image_path)
        self.net.eval()
        self.net.cuda()
        # Sample random latent from Gaus. distribution and produce image via Stylegan
        # then invert it to get the identity latent code
        if random_identity:
            target_path = str(np.random.randint(1000, 9999)) + ".png"
            z = torch.from_numpy(np.random.randn(1, 512)).cuda().float()
            w = self.net.decoder.style(z)
            w = w.unsqueeze(0)
            w_plus = w.repeat(1, 18, 1)
            restyle_latent = self.invert(
                target_path,
                True,
                tensor2im(
                    self.net.decoder(
                        [w_plus],
                        input_is_latent=True,
                        randomize_noise=True,
                        return_latents=True,
                    )[0].squeeze(0)
                ),
            )
            restyle_latent = restyle_latent.squeeze(0)
        elif latent_path is not None:
            with open(latent_path, "rb") as f:
                restyle_latent = torch.from_numpy(pickle.load(f))
        else:
            restyle_latent = self.invert(target_path, aligned, None)

        if not target_path:
            if latent_path:
                target_path = latent_path[:-4] + ".png"

        source_image = Image.open(source_path)
        source_image = source_image.convert("RGB")
        im_transform = get_transforms()
        source_image = im_transform(source_image)
        with torch.no_grad():
            source_cuda = source_image.cuda().float().unsqueeze(0)
            restyle_latent_cuda = restyle_latent.cuda().float().unsqueeze(0)
            tic = time.time()
            result_batch = self.net(
                source_cuda,
                restyle_latent_cuda,
                randomize_noise=True,
                resize=self.opts.resize_outputs,
            )
            toc = time.time()
            print("Inference time:", toc - tic, "s", sep=" ")
        result = tensor2im(result_batch[0])
        crop_size = (0, 240, 1024, 1024)
        result = result.crop(crop_size)
        resize_amount = (1024, (1024 - crop_size[1]))
        if coupled:
            source = Image.open(source_path).crop(crop_size)
            # produce identity(target) image
            latent_image = tensor2im(
                self.net.decoder(
                    [restyle_latent_cuda],
                    input_is_latent=True,
                    randomize_noise=True,
                    return_latents=False,
                )[0][0]
            ).crop(crop_size)
            res = np.concatenate(
                [
                    np.array(source.resize(resize_amount)),
                    np.array(result.resize(resize_amount)),
                    np.array(latent_image.resize(resize_amount)),
                ],
                axis=1,
            )
        else:
            res = np.array(result.resize(resize_amount))

        Image.fromarray(res).save(
            os.path.join(
                self.opts.result_path,
                os.path.basename(source_path)[:-4]
                + "-"
                + os.path.basename(target_path),
            )
        )

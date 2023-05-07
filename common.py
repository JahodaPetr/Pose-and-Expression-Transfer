from PIL import Image
from torchvision.transforms import transforms
import dlib
import numpy as np
import os
import urllib.request
import scipy.ndimage
import bz2
import torch

"""
code modified from https://github.com/eladrich/pixel2style2pixel, https://github.com/yuval-alaluf/restyle-encoder and https://github.com/NVlabs/ffhq-dataset
"""


def tensor2im(var):
    """
    Transforms tensor to image
    """
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = (var + 1) / 2
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype("uint8"))


def get_transforms():
    """
    Returns transforms for images
    """
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def unpack_bz2(src_path):
    """
    Helper function for unpacking facial landmark detector
    """
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, "wb") as fp:
        fp.write(data)
    return dst_path


def get_average_image(net, opts):
    """
    Get average image sampled from latent space
    """
    avg_image = net(
        net.latent_avg.unsqueeze(0),
        input_code=True,
        randomize_noise=False,
        return_latents=False,
        average_code=True,
    )[0]
    avg_image = avg_image.to("cuda").float().detach()
    return avg_image


def run_on_batch(inputs, net, opts, avg_image):
    """
    Runs Restyle inference
    """
    y_hat, latent = None, None
    results_batch = {idx: [] for idx in range(inputs.shape[0])}
    results_latent = {idx: [] for idx in range(inputs.shape[0])}
    for iter in range(opts.n_iters_per_batch):
        if iter == 0:
            avg_image_for_batch = avg_image.unsqueeze(0).repeat(
                inputs.shape[0], 1, 1, 1
            )
            x_input = torch.cat([inputs, avg_image_for_batch], dim=1)
        else:
            x_input = torch.cat([inputs, y_hat], dim=1)

        y_hat, latent = net.forward(
            x_input,
            latent=latent,
            randomize_noise=False,
            return_latents=True,
            resize=opts.resize_outputs,
        )
        # store intermediate outputs
        for idx in range(inputs.shape[0]):
            results_batch[idx].append(y_hat[idx])
            results_latent[idx].append(latent[idx].cpu().numpy())

        y_hat = net.face_pool(y_hat)

    return results_batch, results_latent


def get_landmark_npy(img_path, return_none_with_no_face=False):
    """get landmarks with dlib
    :return: np.array shape=(68, 2)
    """

    if not os.path.isfile("./pretrained_models/shape_predictor_68_face_landmarks.dat"):
        LANDMARKS_MODEL_URL = (
            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        )
        urllib.request.urlretrieve(
            LANDMARKS_MODEL_URL, "shape_predictor_68_face_landmarks.dat.bz2"
        )
        unpack_bz2("pretrained_models/shape_predictor_68_face_landmarks.dat.bz2")
    predictor = dlib.shape_predictor("./pretrained_models/shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    img = dlib.load_rgb_image(img_path)
    dets = detector(img, 1)
    if len(dets) == 0:
        if return_none_with_no_face:
            return None
        else:
            raise RuntimeError("No faces found")

    d = dets[0]
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, d)

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    # lm is a shape=(68,2) np.array
    return lm, img


def align_face_npy_with_params(img_path, output_size=1024, return_none_with_no_face=False):
    """
    Face alignment to match the StyleGANs requirements
    """
    lm, img = get_landmark_npy(
        img_path, return_none_with_no_face=return_none_with_no_face
    )
    if return_none_with_no_face and lm is None:
        return None, None

    lm_chin = lm[0:17]  # left-right
    lm_eyebrow_left = lm[17:22]  # left-right
    lm_eyebrow_right = lm[22:27]  # left-right
    lm_nose = lm[27:31]  # top-down
    lm_nostrils = lm[31:36]  # top-down
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    lm_mouth_inner = lm[60:68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = Image.fromarray(img)

    transform_size = 4096
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (
            int(np.rint(float(img.size[0]) / shrink)),
            int(np.rint(float(img.size[1]) / shrink)),
        )
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1]),
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # # Pad.
    pad = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - img.size[0] + border, 0),
        max(pad[3] - img.size[1] + border, 0),
    )
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(
            np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect"
        )
        actual_padding = pad
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]),
        )
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(
            mask * 3.0 + 1.0, 0.0, 1.0
        )
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), "RGB")
        quad += pad[:2]

    # # Transform.
    img = img.transform(
        (transform_size, transform_size),
        Image.QUAD,
        (quad + 0.5).flatten(),
        Image.BILINEAR,
    )
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    # Save aligned image.
    return np.array(img)

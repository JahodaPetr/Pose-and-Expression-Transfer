from torch.utils.data import Dataset
from PIL import Image
import data_utils
import torch
import pickle
import numpy as np
import os


class ImagesDataset(Dataset):
    def __init__(self, source_root, latents_path, transforms, train=True):
        self.source_paths = sorted(data_utils.make_dataset_dir(source_root))
        self.latents_path = latents_path
        self.transforms = transforms
        self.train = train

    def __len__(self):
        return len(self.source_paths)

    def get_image_pair(self, index):
        from_path = self.source_paths[index]

        folder = os.path.normpath(from_path).split(os.path.sep)[-3]
        n = np.random.randint(0, 10)
        root = os.path.join(self.latents_path, folder, str(n), "latents.pkl")

        with open(root, "rb") as f:
            restyle_latent = torch.from_numpy(pickle.load(f))

        from_im = Image.open(from_path)
        from_im = from_im.convert("RGB")
        from_im = self.transforms(from_im)

        to_im = from_im

        return from_im, to_im, restyle_latent

    def __getitem__(self, index):

        from_im, to_im, restyle_latent = self.get_image_pair(index)
        if self.train:
            new_index = np.random.randint(0, self.__len__())
        else:
            new_index = (index + 10) % self.__len__()
        from_im1, to_im1, restyle_latent1 = self.get_image_pair(new_index)
        return from_im, to_im, restyle_latent, from_im1, to_im1, restyle_latent1

import os
import sys

sys.path.append(".")
sys.path.append("..")

from torch import nn
from torch.utils.data import DataLoader

from common import get_transforms
from paths_config import dataset_paths
from criteria import id_loss, cosface_loss
from images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from psp import MypSp
from training.ranger import Ranger

import torch.nn.functional as F
import torch
"""
code modified from https://github.com/eladrich/pixel2style2pixel

For training ArcFace model weights and StyleGAN2 model weights need to be placed into the pretrained_models directory
ArcFace weights:   https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view
StyleGAN2 weights: https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view
"""


class Coach:
    def __init__(self, opts):
        self.opts = opts
        self.global_step = 0

        self.device = "cuda:0"
        self.opts.device = self.device

        # Initialize network
        self.net = MypSp(self.opts).to(self.device)

        self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type="alex").to(self.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
        if self.opts.cosface_lambda > 0:
            self.cosface_loss = cosface_loss.CosFace(s=5, m=2)
        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.validation_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.opts.batch_size,
            shuffle=True,
            num_workers=int(self.opts.workers),
            drop_last=True,
        )
        self.val_dataloader = DataLoader(
            self.validation_dataset,
            batch_size=self.opts.test_batch_size,
            shuffle=False,
            num_workers=int(self.opts.test_workers),
            drop_last=False,
        )

        self.checkpoint_dir = os.path.join(opts.exp_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

    def train(self):
        self.net.train()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                x, y, restyle_latent, x1, y1, restyle_latent1 = batch
                x, y, restyle_latent = (
                    x.to(self.device).float(),
                    y.to(self.device).float(),
                    restyle_latent.to(self.device).float(),
                )
                # Encoder code so that we dont have to compute it again in the cosface loss
                y_hat, latent, encoder_code = self.net.forward(
                    x, restyle_latent, return_latents=True, return_code=True
                )

                x1, y1, restyle_latent1 = (
                    x1.to(self.device).float(),
                    y1.to(self.device).float(),
                    restyle_latent1.to(self.device).float(),
                )
                # We dont need to calculate another encoder code, because we use the same expression code
                y_hat1, latent1 = self.net.forward(
                    torch.reshape(encoder_code, [x.size()[0], 18, 512]),
                    restyle_latent1,
                    return_latents=True,
                    input_code=True,
                )
                loss, loss_dict, id_logs = self.calc_loss(
                    x,
                    y,
                    y_hat,
                    latent,
                    x1,
                    y1,
                    y_hat1,
                    latent1,
                    encoder_code,
                    restyle_latent1,
                )
                loss.backward()
                self.optimizer.step()

                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix="train")

                # Validation related
                val_loss_dict1 = None
                if self.global_step != 0:
                    if (
                        self.global_step % self.opts.val_interval == 0
                        or self.global_step == self.opts.max_steps
                    ):
                        val_loss_dict1 = self.validate()
                        if val_loss_dict1 and (
                            self.best_val_loss is None
                            or val_loss_dict1["loss"] < self.best_val_loss
                        ):
                            self.best_val_loss = val_loss_dict1["loss"]
                            self.checkpoint_me(
                                val_loss_dict1,
                                y_hat,
                                y,
                                y_hat1,
                                restyle_latent1,
                                is_best=True,
                            )

                    if (
                        self.global_step % self.opts.save_interval == 0
                        or self.global_step == self.opts.max_steps
                    ):
                        if val_loss_dict1 is not None:
                            self.checkpoint_me(
                                val_loss_dict1,
                                y_hat,
                                y,
                                y_hat1,
                                restyle_latent1,
                                is_best=False,
                            )
                        else:
                            self.checkpoint_me(
                                loss_dict,
                                y_hat,
                                y,
                                y_hat1,
                                restyle_latent1,
                                is_best=False,
                            )

                if self.global_step == self.opts.max_steps:
                    print("OMG, finished training!")
                    break

                self.global_step += 1

    def validate(self):
        self.net.eval()
        agg_loss_dict1 = []
        for batch_idx, batch in enumerate(self.val_dataloader):
            x, y, restyle_latent, x1, y1, restyle_latent1 = batch

            with torch.no_grad():
                x, y, restyle_latent, x1, y1, restyle_latent1 = batch
                x, y, restyle_latent = (
                    x.to(self.device).float(),
                    y.to(self.device).float(),
                    restyle_latent.to(self.device).float(),
                )
                y_hat, latent, encoder_code = self.net.forward(
                    x, restyle_latent, return_latents=True, return_code=True
                )
                x1, y1, restyle_latent1 = (
                    x1.to(self.device).float(),
                    y1.to(self.device).float(),
                    restyle_latent1.to(self.device).float(),
                )
                y_hat1, latent1 = self.net.forward(
                    torch.flatten(encoder_code, start_dim=1),
                    restyle_latent1,
                    return_latents=True,
                    input_code=True,
                )
                loss, curr_loss_dict, id_logs = self.calc_loss(
                    x,
                    y,
                    y_hat,
                    latent,
                    x1,
                    y1,
                    y_hat1,
                    latent1,
                    encoder_code,
                    restyle_latent1,
                )

            agg_loss_dict1.append(curr_loss_dict)

        loss_dict1 = data_utils.aggregate_loss_dict(agg_loss_dict1)
        self.print_metrics(loss_dict1, prefix="val")

        self.net.train()
        return loss_dict1

    def checkpoint_me(self, loss_dict, y_hat, y, y_hat1, restyle_latent1, is_best):
        save_name = "best_model.pt" if is_best else f"iteration_{self.global_step}.pt"
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, "timestamp.txt"), "a") as f:
            if is_best:
                f.write(
                    f"**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n"
                )
            else:
                f.write(f"Step - {self.global_step}, \n{loss_dict}\n")

    def configure_optimizers(self):
        params = list(self.net.encoder.parameters()) + list(
            self.net.mapping_net.parameters()
        )
        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())
        if self.opts.optim_name == "adam":
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        train_dataset = ImagesDataset(
            source_root=dataset_paths["face_train"],
            transforms=get_transforms(),
            latents_path=dataset_paths["face_train_latents"],
            train=True,
        )
        validation_dataset = ImagesDataset(
            source_root=dataset_paths["face_val"],
            transforms=get_transforms(),
            latents_path=dataset_paths["face_val_latents"],
            train=False,
        )

        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(validation_dataset)}")
        return train_dataset, validation_dataset

    def calc_loss(
        self,
        x,
        y,
        y_hat,
        latent,
        x1,
        y1,
        y_hat1,
        latent1,
        encoder_code,
        restyle_latent1,
    ):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.opts.cosface_lambda > 0:
            zs_d = self.net.encoder(y_hat)
            zss_d = self.net.encoder(y_hat1)
            zd = encoder_code
            zds = self.net.encoder(x1)
            loss_cosface = self.cosface_loss(zs_d, zss_d, zd, zds)
            loss_dict["loss_cosface"] = float(loss_cosface * self.opts.cosface_lambda)
            loss += loss_cosface * self.opts.cosface_lambda
        if self.opts.id_lambda > 0:
            loss_id, id_logs = self.id_loss(
                y_hat1,
                self.net.face_pool(
                    self.net.decoder(
                        [restyle_latent1],
                        input_is_latent=True,
                        randomize_noise=True,
                        return_latents=False,
                    )[0]
                ),
                x1,
            )
            loss_dict["loss_id"] = float(loss_id * self.opts.id_lambda)
            loss += loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict["loss_l2"] = float(loss_l2 * self.opts.l2_lambda)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict["loss_lpips"] = float(loss_lpips * self.opts.lpips_lambda)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.lpips_lambda_crop > 0:
            loss_lpips_crop = self.lpips_loss(
                y_hat[:, :, 55:243, 32:220], y[:, :, 55:243, 32:220]
            )
            loss_dict["loss_lpips_crop"] = float(
                loss_lpips_crop * self.opts.lpips_lambda_crop
            )
            loss += loss_lpips_crop * self.opts.lpips_lambda_crop
        if self.opts.l2_lambda_crop > 0:
            loss_l2_crop = F.mse_loss(
                y_hat[:, :, 55:243, 32:220], y[:, :, 55:243, 32:220]
            )
            loss_dict["loss_l2_crop"] = float(loss_l2_crop * self.opts.l2_lambda_crop)
            loss += loss_l2_crop * self.opts.l2_lambda_crop

        loss_dict["loss"] = float(loss)
        return loss, loss_dict, id_logs

    def print_metrics(self, metrics_dict, prefix):
        if self.global_step == 0:
            with open(os.path.join(self.checkpoint_dir, f"{prefix}loss.txt"), "w"):
                pass
        print(f"Metrics for {prefix}, step {self.global_step}")
        for key, value in metrics_dict.items():
            print(f"\t{key} = ", value)
            if key == "loss":
                with open(
                    os.path.join(self.checkpoint_dir, f"{prefix}loss.txt"), "a+"
                ) as f:
                    f.write(str(value) + "\n")
        sys.stdout.flush()

    def __get_save_dict(self):
        save_dict = {"state_dict": self.net.state_dict(), "opts": vars(self.opts)}
        return save_dict

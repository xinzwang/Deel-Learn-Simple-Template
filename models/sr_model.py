import logging
from collections import OrderedDict
import random

import torch
import torch.nn as nn
from kornia.color import rgb_to_grayscale

from utils.registry import MODEL_REGISTRY
from utils.process_utils import ShuffleBuffer, calculate_gan_loss_D, calculate_gan_loss_G, crop_test


from .base_model import BaseModel

logger = logging.getLogger("base")


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.data_names = ["src", "tgt"]    # src: Source Image; tgt: Target Image

        self.network_names = ["netSR", "netD"]
        self.networks = {}

        # pix: Pixel Loss; adv: Adversarial Loss; percep: Perceptual Loss
        self.loss_names = ["sr_pix", "sr_adv", "sr_percep"]
        self.loss_weights = {}
        self.losses = {}
        self.optimizers = {}

        # define networks and load pretrained models
        nets_opt = opt["networks"]
        defined_network_names = list(nets_opt.keys())
        assert set(defined_network_names).issubset(set(self.network_names))
        
        for name in defined_network_names:
            setattr(self, name, self.build_network(nets_opt[name]))
            self.networks[name] = getattr(self, name)

        # define variable for training stage
        if self.is_train:
            train_opt = opt["train"]
            # setup loss, optimizers, schedulers
            self.setup_train(opt["train"])

            self.max_grad_norm = train_opt["max_grad_norm"]

            # setup buffers for GAN training
            if self.losses.get("sr_adv"):
                self.D_ratio = train_opt["D_ratio"]
                self.fake_hr_buffer = ShuffleBuffer(train_opt["buffer_size"])
        
    def feed_data(self, data):
        self.src = data["src"] / 255.0      # [0, 255]->[0, 1]
        self.tgt = data["tgt"].to(self.device) / 255.0

    def optimize_parameters(self, step):
        loss_dict = OrderedDict()
        loss_all = 0

        # optimize netSR
        self.sr = self.netSR(self.src)

        if self.losses.get("sr_pix"):
            loss_sr_pixel = self.losses.get("sr_pix")(self.tgt, self.sr)
            loss_dict["sr_pixel"] = loss_sr_pixel.item()
            loss_all += loss_sr_pixel * self.loss_weights["sr_pix"]

        if self.losses.get("sr_percep"):
            loss_sr_percep, loss_sr_style = self.losses.get("sr_percep")(self.tgt, self.sr)
            loss_dict["sr_percep"] = loss_sr_percep.item()
            loss_all += loss_sr_percep * self.loss_weights["sr_percep"]
            if loss_sr_style is not None:
                loss_dict["sr_style"] = loss_sr_style.item()
                loss_all += loss_sr_style * self.loss_weights["sr_percep"]
        
        if self.losses.get("sr_adv"):
            self.set_requires_grad(["netD"], False)
            loss_sr_adv = calculate_gan_loss_G(self.netD, self.losses["sr_adv"], self.tgt, self.sr)
            loss_dict["sr_adv"] = loss_sr_adv.item()
            loss_all += loss_sr_adv * self.loss_weights["sr_adv"]

        self.set_optimizer(names=["netSR"], operation="zero_grad")
        loss_all.backward()
        self.clip_grad_norm(["netSR"], self.max_grad_norm)
        self.set_optimizer(names=["netSR"], operation="step")

        # optimize netD
        if self.losses.get("sr_adv"):
            if step % self.D_ratio == 0:
                self.set_requires_grad(["netD"], True)
                loss_d = calculate_gan_loss_D(self.netD, self.losses["sr_adv"], self.tgt, self.fake_hr_buffer.choose(self.sr))
                loss_dict["d_adv"] = loss_d.item()
                loss_D = loss_d * self.loss_weights["sr_adv"]

                self.set_optimizer(["netD"], operation="zero_grad")
                loss_D.backward()
                self.clip_grad_norm(["netD"], self.max_grad_norm)
                self.set_optimizer(["netD"], "step")

        self.log_dict = loss_dict

    def test(self, test_data, crop_size=None):
        self.src = test_data["src"].to(self.device) / 255.0
        if test_data.get("tgt") is not None:
            self.tgt = test_data["tgt"].to(self.device) / 255.0

        self.set_network_state(["netSR"], "eval")
        with torch.no_grad():
            if crop_size is None:
                self.test_sr = self.netSR(self.src)
            else:
                self.test_sr = crop_test(self.netSR, self.src, self.opt["scale"], crop_size, self.device)
        self.set_network_state(["netSR"], "train")

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lr"] = self.src.detach()[0].float().cpu()
        out_dict["sr"] = self.test_sr.detach()[0].float().cpu()
        return out_dict
    
    
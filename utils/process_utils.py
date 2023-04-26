import random
import torch


### Utils for training GAN 
class ShuffleBuffer():
    """Random choose previous generated images or ones produced by the latest generators.
    :param buffer_size: the size of image buffer
    :type buffer_size: int
    """

    def __init__(self, buffer_size):
        """Initialize the ImagePool class.
        :param buffer_size: the size of image buffer
        :type buffer_size: int
        """
        self.buffer_size = buffer_size
        self.num_imgs = 0
        self.images = []

    def choose(self, images, prob=0.5):
        """Return an image from the pool.
        :param images: the latest generated images from the generator
        :type images: list
        :param prob: probability (0~1) of return previous images from buffer
        :type prob: float
        :return: Return images from the buffer
        :rtype: list
        """
        if self.buffer_size == 0:
            return  images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.buffer_size:
                self.images.append(image)
                return_images.append(image)
                self.num_imgs += 1
            else:
                p = random.uniform(0, 1)
                if p < prob:
                    idx = random.randint(0, self.buffer_size - 1)
                    stored_image = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(stored_image)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


def calculate_gan_loss_D( netD, criterion, real, fake):
    d_pred_fake = netD(fake.detach())
    d_pred_real = netD(real)
    loss_real = criterion(d_pred_real, True, is_disc=True)
    loss_fake = criterion(d_pred_fake, False, is_disc=True)
    return (loss_real + loss_fake) / 2.0
    
def calculate_gan_loss_G(netD, criterion, real, fake):
    d_pred_fake = netD(fake)
    loss_real = criterion(d_pred_fake, True, is_disc=False)
    return loss_real


### Utils for low-level CV Test
def crop_test(netSR, lr, scale, crop_size, device):
        b, c, h, w = lr.shape

        h_start = list(range(0, h-crop_size, crop_size))
        w_start = list(range(0, w-crop_size, crop_size))

        sr1 = torch.zeros(b, c, int(h*scale), int(w* scale), device=device) - 1
        for hs in h_start:
            for ws in w_start:
                lr_patch = lr[:, :, hs: hs+crop_size, ws: ws+crop_size]
                sr_patch = netSR(lr_patch)

                sr1[:, :, 
                    int(hs*scale):int((hs+crop_size)*scale),
                    int(ws*scale):int((ws+crop_size)*scale)
                ] = sr_patch
        
        h_end = list(range(h, crop_size, -crop_size))
        w_end = list(range(w, crop_size, -crop_size))

        sr2 = torch.zeros(b, c, int(h*scale), int(w* scale), device=device) - 1
        for hd in h_end:
            for wd in w_end:
                lr_patch = lr[:, :, hd-crop_size:hd, wd-crop_size:wd]
                sr_patch = netSR(lr_patch)

                sr2[:, :, 
                    int((hd-crop_size)*scale):int(hd*scale),
                    int((wd-crop_size)*scale):int(wd*scale)
                ] = sr_patch

        mask1 = (
            (sr1 == -1).float() * 0 + 
            (sr2 == -1).float() * 1 + 
            ((sr1 > 0) * (sr2 > 0)).float() * 0.5
        )

        mask2 = (
            (sr1 == -1).float() * 1 + 
            (sr2 == -1).float() * 0 + 
            ((sr1 > 0) * (sr2 > 0)).float() * 0.5
        )

        sr = mask1 * sr1 + mask2 * sr2

        return sr

from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt

class ImgDataset(Dataset):
    def __init__(self, data_path, data_start, data_end, permu):
        self.data_path = data_path
        self._load_imgs(data_path, data_start, data_end)
        self.data_length = self.img_names.shape[0]
        self.img_preprocess = transforms.Compose([
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ])
        self.permu = permu

    def __len__(self):
        return self.data_length

    def _load_imgs(self, path, data_start, data_end):
        names = os.listdir(path)
        names.sort()
        assert len(names) == 5000
        self.img_names = np.array([str(names[i]) for i in range(data_start, data_end)], dtype=np.string_)

    def __getitem__(self, index):
        img_id = self.permu[index]
        img_path = os.path.join(self.data_path, self.img_names[img_id].decode())
        img = Image.open(img_path)
        img = self.img_preprocess(img)
        return img


def vmap(func, M):
    # from https://discuss.pytorch.org/t/a-fast-way-to-apply-a-function-across-an-axis/8378
    tList = [func(m) for m in torch.unbind(M, dim=0)]
    res = torch.stack(tList, dim=0)

    return res

class GaussianHistogram(nn.Module):
    # from https://discuss.pytorch.org/t/differentiable-torch-histc/25865/2
    def __init__(self, bins, min, max, sigma=0.003, device='cuda:0'):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float().to(device) + 0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        x = x.sum(dim=1) / (512*512)
        return x

class Img:
    def __init__(self, resize=True):
        super(Img, self).__init__()
        if resize:
            self.img_preprocess = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
            ])
        else:
            self.img_preprocess = transforms.Compose([
                transforms.ToTensor(),
            ])

    def read_img(self, path):
        img = Image.open(path)
        img = self.img_preprocess(img)
        return img

    @classmethod
    def write_img(cls, img, path):
        assert img.max() <= 1.0 and img.min() >= 0.0
        save_image(img, path)

    @classmethod
    def apply_lut(cls, img_batch, lut, device):
        # img_batch: batchsize x 3 channels x H x W
        # lut: batchsize x 3 channels x 256
        assert lut.shape[1] == 3 and lut.shape[2] == 256
        img_IntTensor = (img_batch * 255).type(torch.LongTensor).to(device)
        out_img_batchs = []
        for b in range(img_batch.shape[0]):
            out_img_chans = []
            for i in range(3):
                out_img_chan = torch.index_select(lut[b, i, :], dim=0, index=img_IntTensor[b, i, :, :].reshape(-1))
                out_img_chans.append(out_img_chan.reshape(img_batch[b, i, :, :].shape) / 255.)
            out_img_chans = torch.stack(out_img_chans, dim=0)
            out_img_batchs.append(out_img_chans)
        out_img_batchs = torch.stack(out_img_batchs, dim=0)
        return out_img_batchs

    @classmethod
    def interp_lut(cls, lut_raw):
        # lut_raw: batchsize x 3 channels x lut_dim
        lut_intp_chans = []
        for i in range(3):
            lut_intp = F.interpolate(lut_raw[:, i, :].unsqueeze(1).unsqueeze(1),
                                     (1, 256),
                                     mode='bicubic',
                                     align_corners=True).clamp(min=0,max=255).squeeze(1).squeeze(1)
            lut_intp_chans.append(lut_intp)
        lut_intp_chans = torch.stack(lut_intp_chans, dim=1)
        # lut_intp_chans: batchsize x 3 channels x 256
        return lut_intp_chans

    @classmethod
    def color_hist(cls, img_batch, histc):
        # image_batch: batchsize x 3 channels x H x W
        def color_hist_single(img_batch):
            out = []
            for i in range(img_batch.shape[0]):
                out.append(histc(img_batch[i, :, :].flatten()))
            return torch.stack(out, dim=0)
        return vmap(color_hist_single, img_batch)

    @classmethod
    def color_hist_torch(cls, img_batch):
        # image_batch: batchsize x 3 channels x H x W
        def color_hist_single(img_batch):
            out = []
            for i in range(img_batch.shape[0]):
                out.append(torch.histc(img_batch[i, :, :].flatten(), bins=256, min=0, max=1))
            return torch.stack(out, dim=0)
        return vmap(color_hist_single, img_batch)

if __name__ == '__main__':
    import time
    ta = time.time()
    dataset_test = ImgDataset('fivek_dataset/processed/lut_0', 0, 5000, np.arange(5000))
    dataloader = DataLoader(dataset_test, batch_size=64,
                            shuffle=False, num_workers=0)
    print(time.time() - ta)
    ta = time.time()
    for i_batch, sample_batched in enumerate(dataloader):
        print(sample_batched.shape)
        print(time.time() - ta)
        ta = time.time()
        if i_batch == 1:
            break
    test = Img()
    lut_tensor_1 = torch.Tensor([[0, 64, 130, 192, 255],
                               [0, 64, 128, 192, 255],
                               [0, 64, 180, 192, 255]])
    lut_tensor_2 = torch.Tensor([[0, 64, 130, 192, 255],
                               [0, 64, 180, 192, 255],
                               [0, 64, 128, 192, 255]])
    lut_tensor = torch.stack((lut_tensor_1, lut_tensor_2), dim=0)
    print(lut_tensor.shape)
    lut = test.interp_lut(lut_tensor)
    print(lut.shape)
    fig, ax = plt.subplots(1, 2)
    for i in range(2):
        ax[i].plot(lut[i, 0, :], color='red')
        ax[i].plot(lut[i, 1, :], color='green')
        ax[i].plot(lut[i, 2, :], color='blue')
    plt.savefig('test/lut_curve.png')
    plt.close()
    img = test.read_img('test/apple.jpg')
    out_img = test.apply_lut(torch.stack((img, img), dim=0), lut, 'cpu')
    test.write_img(out_img[0, :, :, :], 'test/apple_out_0.jpg')
    test.write_img(out_img[1, :, :, :], 'test/apple_out_1.jpg')
    img = test.read_img('lut_test/city.jpg').to('cuda:0')
    histc = GaussianHistogram(bins=64, min=0, max=1)
    hists = Img.color_hist(img.unsqueeze(0), histc).cpu().numpy()
    print(hists.shape)
    plt.bar(np.arange(64), hists[0, 0, :])
    plt.savefig('test/hist.png')

import argparse
import torch
import matplotlib.pyplot as plt

from dataset import Img
from train import init_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lut_points', type=int, default=32,
                        help='lut points')
    parser.add_argument('--load_from', type=str, default="newest.ckpt",
                        help='load from ckpt')
    parser.add_argument('--give_ref', type=bool, default=False,
                        help='give reference img and color graded img')
    parser.add_argument('--lut', type=str, default=None,
                        help='reference lut img')
    parser.add_argument('--input', type=str, default=None,
                        help='input img')
    parser.add_argument('--input_type', type=str, default='img',
                        help='input type')
    parser.add_argument('--clear_lut', type=bool, default=False,
                        help='clear LUT instead of apply')
    parser.add_argument('--output_lut', type=str, default='lut_test/lut.png',
                        help='output lut path')
    parser.add_argument('--output_path', type=str, default=None,
                        help='output img path')
    args = parser.parse_args()

    model, device = init_model(args)
    model.eval()

    Img_resize = Img()
    Img_noresize = Img(False)
    if not args.clear_lut:
        lut_img = Img_resize.read_img(args.lut).unsqueeze(0).to(device)
    input_img = Img_resize.read_img(args.input).unsqueeze(0).to(device)
    with torch.no_grad():
        if args.clear_lut:
            output_lut = model(input_img)
        else:
            output_lut = model(lut_img)
        output_interp_lut = Img.interp_lut(output_lut)
        input_img = Img_noresize.read_img(args.input).unsqueeze(0).to(device)
        output_batch = Img.apply_lut(input_img, output_interp_lut, device)
    Img.write_img(output_batch[0, :, :, :], args.output_path)

    lut = output_interp_lut[0, :, :].detach().cpu().numpy()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(lut[0, :], color='red')
    ax.plot(lut[1, :], color='green')
    ax.plot(lut[2, :], color='blue')
    plt.savefig(args.output_lut)
    plt.close()


if __name__ == '__main__':
    main()

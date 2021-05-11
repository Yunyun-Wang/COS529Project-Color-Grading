import os
import argparse
import numpy as np
import torch
import time
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from model import LutGenerator, LutGeneratorHist
from dataset import ImgDataset, Img, GaussianHistogram

def lut_to_fig(lut, writer, l, i):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(lut[0, :], color='red')
    ax.plot(lut[1, :], color='green')
    ax.plot(lut[2, :], color='blue')
    writer.add_figure("lut-{}".format(l), fig, i)
    plt.close()

def hist_to_fig(hist1, hist2, writer, l, i):
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(hist1[0, :], color='red')
    ax[0].plot(hist1[1, :], color='green')
    ax[0].plot(hist1[2, :], color='blue')
    ax[1].plot(hist2[0, :], color='red')
    ax[1].plot(hist2[1, :], color='green')
    ax[1].plot(hist2[2, :], color='blue')
    writer.add_figure("hist-{}".format(l), fig, i)
    plt.close()

def save_ckpt(model, exp_name, step):
    torch.save({'model': model.state_dict()}, "{}/newest.ckpt".format(exp_name))
    torch.save({'model': model.state_dict()}, "{}/step-{}.ckpt".format(exp_name, step))
    print("Checkpoint saved!")


def load_ckpt(model, device, load_from):
    if os.path.exists(load_from) == False:
        print("Checkpoint not exist!")
        return
    ckpt = torch.load(load_from, map_location=device)
    model.load_state_dict(ckpt['model'])
    print("Checkpoint loaded!")


def train(args, model, writer, device):
    print("Loading data...")
    permu = np.random.permutation(args.split)
    original_train = ImgDataset('fivek_dataset/processed/original_small', 0, args.split, permu)
    load_original_train = DataLoader(original_train, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True)
    iter_original_train = iter(load_original_train)
    lut_train = []
    load_lut_train = []
    iter_lut_train = []
    for i in range(args.lut_num):
        lut_train.append(ImgDataset('fivek_dataset/processed/lut_{}'.format(i), 0, args.split, permu))
        load_lut_train.append(DataLoader(lut_train[-1], batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True))
        iter_lut_train.append(iter(load_lut_train[-1]))
    if args.input_type == 'img':
        histc = GaussianHistogram(bins=64, min=0, max=1).to(device)

    print("Training...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    losses = []
    ta = time.time()
    for i in range(args.iters):
        try:
            org_imgs_1 = next(iter_original_train).to(device)
        except:
            iter_original_train = iter(load_original_train)
            org_imgs_1 = next(iter_original_train).to(device)
        try:
            org_imgs_2 = next(iter_original_train).to(device)
        except:
            iter_original_train = iter(load_original_train)
            org_imgs_2 = next(iter_original_train).to(device)
        for l in range(args.lut_num):
            try:
                lut_imgs_1 = next(iter_lut_train[l]).to(device)
            except:
                iter_lut_train[l] = iter(load_lut_train[l])
                lut_imgs_1 = next(iter_lut_train[l]).to(device)
            try:
                lut_imgs_2 = next(iter_lut_train[l]).to(device)
            except:
                iter_lut_train[l] = iter(load_lut_train[l])
                lut_imgs_2 = next(iter_lut_train[l]).to(device)

            if args.give_ref:
                input_batch = torch.cat((org_imgs_1, lut_imgs_1), dim=1)
            else:
                input_batch = lut_imgs_1
            if args.input_type == 'hist':
                input_batch = Img.color_hist_torch(input_batch)
            optimizer.zero_grad()

            output_lut = model(input_batch)
            output_interp_lut = Img.interp_lut(output_lut)

            if args.clear_lut:
                output_batch = Img.apply_lut(lut_imgs_1, output_interp_lut, device)
            else:
                output_batch = Img.apply_lut(org_imgs_2, output_interp_lut, device)

            if args.loss_type == 'l2':
                if args.clear_lut:
                    loss = criterion(output_batch, org_imgs_1)
                else:
                    loss = criterion(output_batch, lut_imgs_2)
            else:
                hist_now = Img.color_hist(output_batch, histc)
                hist_target = Img.color_hist(lut_imgs_2, histc)
                loss = criterion(hist_now, hist_target) + criterion(output_batch, lut_imgs_2)

            loss.backward()
            optimizer.step()

            if i % args.log_step == 0:
                if args.clear_lut:
                    writer.add_image("comp-{}".format(l), torch.cat(
                        (lut_imgs_1[0, :, :, :], output_batch[0, :, :, :], org_imgs_1[0, :, :, :]), dim=2), i)
                else:
                    writer.add_image("comp-{}".format(l), torch.cat(
                        (org_imgs_2[0, :, :, :], output_batch[0, :, :, :], lut_imgs_2[0, :, :, :]), dim=2), i)
                lut_to_fig(output_interp_lut[0, :, :].detach().cpu().numpy(), writer, l, i)
                if args.loss_type == 'hist':
                    hist_to_fig(hist_now[0, :, :].detach().cpu().numpy(), hist_target[0, :, :].detach().cpu().numpy(), writer, l, i)

            losses.append(loss.item())
            # print(output_lut.detach().cpu().numpy()[0, :])

        if i % args.val_step == 0:
            pass

        if i % args.log_step == 0:
            # print(torch.argmax(output, dim=1) == target)
            print('Train iter: [{}/{} ({:.0f}%) {:.2f} sec]\tLoss: {:.6f}'.format(
                i + 1, args.iters, 100 * (i + 1) / args.iters, time.time() - ta, np.mean(losses)))
            ta = time.time()
            writer.add_scalar("loss", np.mean(losses), i)
            losses = []
        if (i + 1) % args.save_step == 0:
            save_ckpt(model, args.exp_name, i+1)
    save_ckpt(model, args.exp_name, 'final')

def test_lut(model, args, device, load_original_test, load_lut_test, load_original_test_rand, load_lut_test_rand):
    iter_original_test = iter(load_original_test)
    iter_lut_test = iter(load_lut_test)
    iter_original_test_rand = iter(load_original_test_rand)
    iter_lut_test_rand = iter(load_lut_test_rand)
    losses = []
    original_dists = []
    for i in range(len(load_original_test)):
        org_imgs = next(iter_original_test).to(device)
        lut_imgs = next(iter_lut_test).to(device)
        for j in range(args.test_rand_num):
            try:
                org_imgs_rand = next(iter_original_test_rand).to(device)
                lut_imgs_rand = next(iter_lut_test_rand).to(device)
            except:
                iter_original_test_rand = iter(load_original_test_rand)
                iter_lut_test_rand = iter(load_lut_test_rand)
                org_imgs_rand = next(iter_original_test_rand).to(device)
                lut_imgs_rand = next(iter_lut_test_rand).to(device)

            if args.give_ref:
                input_batch = torch.cat((org_imgs_rand, lut_imgs_rand), dim=1)
            else:
                input_batch = lut_imgs_rand
            if args.input_type == 'hist':
                input_batch = Img.color_hist_torch(input_batch)

            with torch.no_grad():
                output_lut = model(input_batch)
                output_interp_lut = Img.interp_lut(output_lut)
                output_batch = Img.apply_lut(org_imgs, output_interp_lut, device)

            loss = F.mse_loss(output_batch, lut_imgs).item()
            losses.append(loss)
        original_dists.append(F.mse_loss(org_imgs, lut_imgs).item())
    return np.mean(losses), np.mean(original_dists)



def test(model, args, device):
    np.random.seed(233)
    model.eval()
    permu = np.arange(5000-args.split)
    permu_rand = np.random.permutation(5000-args.split)
    original_test = ImgDataset('fivek_dataset/processed/original_small', args.split, 5000, permu)
    load_original_test = DataLoader(original_test, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                     drop_last=True)
    original_test_rand = ImgDataset('fivek_dataset/processed/original_small', args.split, 5000, permu_rand)
    load_original_test_rand = DataLoader(original_test_rand, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                     drop_last=True)

    tot_luts = 35
    avg_loss_list = []
    org_avg_loss_list = []
    for i in range(args.lut_num):
        lut_test = ImgDataset('fivek_dataset/processed/lut_{}'.format(i), args.split, 5000, permu)
        load_lut_test = DataLoader(lut_test, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True)
        lut_test_rand = ImgDataset('fivek_dataset/processed/lut_{}'.format(i), args.split, 5000, permu_rand)
        load_lut_test_rand = DataLoader(lut_test_rand, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True)
        avg_l2, org_avg_l2 = test_lut(model, args, device, load_original_test, load_lut_test, load_original_test_rand, load_lut_test_rand)
        avg_loss_list.append(avg_l2)
        org_avg_loss_list.append(org_avg_l2)
        print('lut {} loss: {} org loss:{}'.format(i, avg_l2, org_avg_l2))

    permu = np.arange(5000)
    permu_rand = np.random.permutation(5000)
    original_test = ImgDataset('fivek_dataset/processed/original_small', 0, 5000, permu)
    load_original_test = DataLoader(original_test, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                    drop_last=True)
    original_test_rand = ImgDataset('fivek_dataset/processed/original_small', 0, 5000, permu_rand)
    load_original_test_rand = DataLoader(original_test_rand, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                    drop_last=True)
    for i in range(args.lut_num, tot_luts):
        lut_test = ImgDataset('fivek_dataset/processed/lut_{}'.format(i), 0, 5000, permu)
        load_lut_test = DataLoader(lut_test, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True)
        lut_test_rand = ImgDataset('fivek_dataset/processed/lut_{}'.format(i), 0, 5000, permu_rand)
        load_lut_test_rand = DataLoader(lut_test_rand, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True)
        avg_l2, org_avg_l2 = test_lut(model, args, device, load_original_test, load_lut_test, load_original_test_rand, load_lut_test_rand)
        avg_loss_list.append(avg_l2)
        org_avg_loss_list.append(org_avg_l2)
        print('lut {} loss: {} org loss:{}'.format(i, avg_l2, org_avg_l2))

    print('total avg loss: {} total org avg loss: {}'.format(np.mean(avg_loss_list), np.mean(org_avg_loss_list)))
    print('seen lut avg loss: {} seen org lut avg loss: {}'.format(np.mean(avg_loss_list[:args.lut_num]), np.mean(org_avg_loss_list[:args.lut_num])))
    print('unseen lut avg loss: {} unseen org lut avg loss: {}'.format(np.mean(avg_loss_list[args.lut_num:]), np.mean(org_avg_loss_list[args.lut_num:])))
    step = args.load_from.split('/')[1]
    step = step.split('-')[1]
    step = int(step.split('.')[0])
    pickle.dump((avg_loss_list, org_avg_loss_list), open('results/give_ref_{}-lut_points_{}-step_{}.pkl'.format(args.give_ref, args.lut_points, step), 'wb'))

def demo(model, args, writer, device):
    demo_num = 5
    histc = GaussianHistogram(bins=64, min=0, max=1).to(device)

    np.random.seed(233)
    permu_rand = np.random.permutation(5000-args.split)
    original_test_rand = ImgDataset('fivek_dataset/processed/original_small', args.split, 5000, permu_rand)
    load_original_test_rand = DataLoader(original_test_rand, batch_size=demo_num, shuffle=False, num_workers=2,
                                         drop_last=True)
    for l in range(args.lut_num):
        lut_test_rand = ImgDataset('fivek_dataset/processed/lut_{}'.format(l), args.split, 5000, permu_rand)
        load_lut_test_rand = DataLoader(lut_test_rand, batch_size=demo_num, shuffle=False, num_workers=2, drop_last=True)
        iter_original_test_rand = iter(load_original_test_rand)
        iter_lut_test_rand = iter(load_lut_test_rand)

        org_imgs_rand_1 = next(iter_original_test_rand).to(device)
        lut_imgs_rand_1 = next(iter_lut_test_rand).to(device)
        org_imgs_rand_2 = next(iter_original_test_rand).to(device)
        lut_imgs_rand_2 = next(iter_lut_test_rand).to(device)

        input_batch = lut_imgs_rand_1
        if args.input_type == 'hist':
            input_batch = Img.color_hist_torch(input_batch)
        with torch.no_grad():
            output_lut = model(input_batch)
            output_interp_lut = Img.interp_lut(output_lut)
            output_batch = Img.apply_lut(org_imgs_rand_2, output_interp_lut, device)

        hist_now = Img.color_hist(output_batch, histc)
        hist_target = Img.color_hist(lut_imgs_rand_2, histc)
        for i in range(demo_num):
            if args.clear_lut:
                writer.add_image("comp-{}".format(l), torch.cat(
                    (lut_imgs_rand_1[i, :, :, :], output_batch[i, :, :, :], org_imgs_rand_1[i, :, :, :]), dim=2), i)
            else:
                writer.add_image("comp-{}".format(l), torch.cat(
                    (org_imgs_rand_2[i, :, :, :], lut_imgs_rand_1[i, :, :, :], output_batch[i, :, :, :], lut_imgs_rand_2[i, :, :, :]), dim=2), i)
            lut_to_fig(output_interp_lut[i, :, :].detach().cpu().numpy(), writer, l, i)
            hist_to_fig(hist_now[i, :, :].detach().cpu().numpy(), hist_target[i, :, :].detach().cpu().numpy(), writer,
                        l, i)




def init_model(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.input_type == 'img':
        generator = LutGenerator
    else:
        generator = LutGeneratorHist
    if args.give_ref:
        model = generator(6, lut_points=args.lut_points).to(device)
    else:
        model = generator(3, lut_points=args.lut_points).to(device)
    load_ckpt(model, device, args.load_from)
    return model, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--iters', type=int, default=1000000,
                        help='iterations')
    parser.add_argument('--save_step', type=int, default=50,
                        help='save every xx steps')
    parser.add_argument('--log_step', type=int, default=10,
                        help='log every xx steps')
    parser.add_argument('--val_step', type=int, default=10,
                        help='val every xx steps')
    parser.add_argument('--split', type=int, default=4000,
                        help='split on train/test')
    parser.add_argument('--total_num', type=int, default=5000,
                        help='total image number')
    parser.add_argument('--lut_num', type=int, default=30,
                        help='total lut number')
    parser.add_argument('--lut_points', type=int, default=32,
                        help='lut points')
    parser.add_argument('--test_rand_num', type=int, default=5,
                        help='random x times during test')
    parser.add_argument('--load_from', type=str, default="newest.ckpt",
                        help='load from ckpt')
    parser.add_argument('--only_test', type=bool, default=False,
                        help='skip training, only test')
    parser.add_argument('--only_demo', type=bool, default=False,
                        help='skip training, only demo')
    parser.add_argument('--give_ref', type=bool, default=False,
                        help='give reference img and color graded img')
    parser.add_argument('--loss_type', type=str, default="l2",
                        help='type of loss l2/hist')
    parser.add_argument('--input_type', type=str, default="img",
                        help='type of input img/hist')
    parser.add_argument('--clear_lut', type=bool, default=False,
                        help='clear LUT instead of apply')
    parser.add_argument('--exp_name', type=str, default="test",
                        help='exp name')
    args = parser.parse_args()

    model, device = init_model(args)
    os.makedirs(args.exp_name, exist_ok=True)

    if not args.only_test and not args.only_demo:
        writer = SummaryWriter(log_dir='log/{}'.format(args.exp_name))
        try:
            train(args, model, writer, device)
        except KeyboardInterrupt:
            save_ckpt(model, args.exp_name, 'final')
    if args.only_demo:
        writer = SummaryWriter(log_dir='log/demo')
        demo(model, args, writer, device)
    else:
        test(model, args, device)


if __name__ == '__main__':
    main()

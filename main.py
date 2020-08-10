import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms
from augmix import AugMixDataset
from WideResNet_pytorch.wideresnet import WideResNet

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def main(args):
    if args.no_jsd:
        ch = 'no_jsd'
    elif args.wo_jsd:
        ch = 'wo_jsd'
    else:
        ch = 'jsd'

    PATH = "./ckpt/t_{}.ckpt".format(ch)
    torch.manual_seed(2020)
    np.random.seed(2020)

    # 1. dataload
    # basic augmentation & preprocessing
    train_base_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4)
    ])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)
    ])
    # train_transform = transforms.Compose(train_base_aug + preprocess)
    test_transform = preprocess
    # load data
    train_data = datasets.CIFAR100('./data/cifar', train=True, transform=train_base_aug, download=True)
    train_data = AugMixDataset(train_data, preprocess, args.no_jsd)

    test_data = datasets.CIFAR100('./data/cifar', train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True)
    # 2. model
    # wideresnet 40-2
    model = WideResNet(depth=40, num_classes=100, widen_factor=2, drop_rate=0.0)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        0.1,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader),
                                                           eta_min=1e-6,
                                                           last_epoch=-1)

    model = nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    # training model with cifar100

    if not args.test:
        model.train()
        losses = []

        for epoch in range(args.epochs):
            pbar = tqdm(train_loader)
            for images, targets in pbar:

                optimizer.zero_grad()

                if args.no_jsd:
                    images, targets = images.cuda(), targets.cuda()
                    logits = model(images)
                    loss = F.cross_entropy(logits, targets)
                    pbar.set_description('Epoch: {:2d} | Loss: {:.4f}'.format(epoch + 1, loss.item()))
                else:
                    N = images[0].size(0)
                    ori_aug1_aug2 = torch.cat(images, dim=0).cuda()
                    targets = targets.cuda()
                    logits = model(ori_aug1_aug2)
                    # logits_o, logits_1, logits_2 = logits[:N], logits[N:2*N], logits[2*N:]
                    logits_o, logits_1, logits_2 = torch.split(logits, N)

                    if not args.wo_jsd:
                        ori_loss = F.cross_entropy(logits_o, targets)
                        jsd = train_data.aug.jensen_shannon(logits_o, logits_1, logits_2)
                        loss = ori_loss + 12 * jsd
                        pbar.set_description(
                            'Epoch: {:2d} | Total Loss: {:.4f} | JSD: {:.4f}'.format(epoch + 1, loss.item(),
                                                                                     jsd.item()))

                    else:
                        loss = F.cross_entropy(logits_o, targets)
                        pbar.set_description('Epoch: {:2d} | Loss: {:.4f}'.format(epoch + 1, loss.item()))

                loss.backward()
                optimizer.step()
                scheduler.step()

                losses.append(loss.item())

            torch.save({
                "epoch": epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses
            }, PATH)
        # evaluate on cifar100-c
        model.eval()
        total_accs = []
        for corruption in CORRUPTIONS:
            accs = []
            test_data.data = np.load('./data/cifar/CIFAR-100-C/%s.npy' % corruption)
            test_data.targets = torch.LongTensor(np.load('./data/cifar/CIFAR-100-C/labels.npy'))

            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True)

            pbar = tqdm(test_loader)
            for images, targets in pbar:
                images, targets = images.cuda(), targets.cuda()

                with torch.no_grad():
                    logits = model(images)

                loss = F.cross_entropy(logits, targets)
                preds = torch.argmax(logits, dim=1)
                acc = torch.tensor((preds == targets), dtype=torch.float32).mean().item()
                accs.append(acc)

            accs = sum(accs) / len(accs)
            total_accs.append(accs)
            print('{} CE: {:.1f}%'.format(corruption, 100 - 100 * accs))

        mCE = 100 - 100 * sum(total_accs) / len(total_accs)
        print('mCE: {:.1f}%'.format(mCE))
    # test
    else:
        checkpoint = torch.load(args.path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        losses = checkpoint['losses']

        print('loadded loss:', sum(losses) / len(losses))

        total_accs = []
        for corruption in CORRUPTIONS:
            accs = []
            test_data.data = np.load('./data/cifar/CIFAR-100-C/%s.npy' % corruption)
            test_data.targets = torch.LongTensor(np.load('./data/cifar/CIFAR-100-C/labels.npy'))

            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True)

            pbar = tqdm(test_loader)
            for images, targets in pbar:
                images, targets = images.cuda(), targets.cuda()

                with torch.no_grad():
                    logits = model(images)
                    preds = torch.argmax(logits, dim=1)
                    acc = torch.tensor((preds == targets), dtype=torch.float32).mean().item()
                    accs.append(acc)

            accs = sum(accs) / len(accs)
            total_accs.append(accs)
            print('{} CE: {:.1f}%'.format(corruption, 100 - 100 * accs))

        mCE = 100 - 100 * sum(total_accs) / len(total_accs)
        print('mCE: {:.1f}%'.format(mCE))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--no_jsd', action='store_true', default=False,
                        help='Calculating loss w/o JSD')

    parser.add_argument('--wo_jsd', action='store_true', default=False,
                        help='augmix w/o jsd loss')

    parser.add_argument('--severity', default=3, type=int,
                        help='Severity of base augmentation operators')

    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--batch_size', default=256, type=int)

    parser.add_argument('--test', action='store_true', default=False)

    parser.add_argument('--path', type=str, default='./ckpt/wrn40-2.ckpt')

    args = parser.parse_args()

    main(args)

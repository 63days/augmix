import torch
import numpy as np
import matplotlib.pyplot as plt
from augmix import AugMix
from torchvision import datasets
from torchvision import transforms


def main():
    preprocess = transforms.ToTensor()

    train_data = datasets.CIFAR100('./data/cifar', train=True, download=True)
    augmix = AugMix()
    img, img_num = train_data[np.random.randint(50000)]
    fig = plt.figure()
    rows = 1
    cols = 2
    mix = torch.zeros_like(preprocess(img))
    w = augmix.dirichlet.sample()
    for i in range(augmix.k):
        aug = img.copy()
        depth = np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(augmix.augs)
            aug = op(aug, augmix.severity)
        mix = mix + w[i] * preprocess(aug)

    m = augmix.beta.sample()
    augmix = m * preprocess(img) + (1 - m) * mix
    augmix = augmix.permute(1, 2, 0)

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(np.asarray(img))
    ax1.set_title('original')
    ax1.axis("off")

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(np.asarray(augmix))
    ax2.set_title('augmix')
    ax2.axis("off")

    plt.savefig('./res/img_{}'.format(img_num))
    plt.show()


if __name__ == '__main__':
    main()

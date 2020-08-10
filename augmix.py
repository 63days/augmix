import torch
import torch.nn as nn
import torch.nn.functional as F
from augmentations import *
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta

#dv = 'cuda' if torch.cuda.is_available() else 'cpu'


class AugMixDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess, no_jsd=False):
        super(AugMixDataset, self).__init__()
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd
        self.aug = AugMix()

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return self.preprocess(x), y
        else:
            aug1 = self.aug.augment_and_mix(x, self.preprocess)
            aug2 = self.aug.augment_and_mix(x, self.preprocess)
            return (self.preprocess(x), aug1, aug2), y

    def __len__(self):
        return len(self.dataset)


class AugMix(nn.Module):
    def __init__(self, k=3, alpha=1, severity=3):
        super(AugMix, self).__init__()
        self.k = k
        self.alpha = alpha
        self.severity = severity
        self.dirichlet = Dirichlet(torch.full(torch.Size([k]), alpha, dtype=torch.float32))
        self.beta = Beta(alpha, alpha)
        self.augs = augmentations
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def augment_and_mix(self, images, preprocess):
        '''
        Args:
            images: PIL Image
            preprocess: transform[ToTensor, Normalize]

        Returns: AugmentAndMix Tensor
        '''
        mix = torch.zeros_like(preprocess(images))
        w = self.dirichlet.sample()
        for i in range(self.k):
            aug = images.copy()
            depth = np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.augs)
                aug = op(aug, 3)
            mix = mix + w[i] * preprocess(aug)

        m = self.beta.sample()

        augmix = m * preprocess(images) + (1 - m) * mix

        return augmix

    def jensen_shannon(self, logits_o, logits_1, logits_2):
        p_o = F.softmax(logits_o, dim=1)
        p_1 = F.softmax(logits_1, dim=1)
        p_2 = F.softmax(logits_2, dim=1)

        # kl(q.log(), p) -> KL(p, q)
        M = torch.clamp((p_o + p_1 + p_2) / 3, 1e-7, 1)  # to avoid exploding
        js = (self.kl(M.log(), p_o) + self.kl(M.log(), p_1) + self.kl(M.log(), p_2)) / 3
        return js



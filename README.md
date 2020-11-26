# AugMix:Pytorch Implementation
Pytorch Implementation of AugMix (ICLR2020)

## What is AugMix?
AugMix is an abbreviation for Augmentation and Mix. In machine learning, augmentation is the technique to increase the diversity of data available for training model by slightly modifying training data. Modern deep neural networks can achieve high accuracy when the training distribution and test distribution are identically distributed, but this assumption is frequently violated in practice. In this paper, authors propose AugMix, a data preprocessing technique which improves model robustness and uncertainty using augmentation.  
![image](https://user-images.githubusercontent.com/37788686/90134108-9eb48c00-ddab-11ea-8732-da1446729f32.png)

## Realization of AugMix

| AugMix | Cascade of successive compositions |
| ------ | ------------ |
| ![image](https://user-images.githubusercontent.com/37788686/90134412-197da700-ddac-11ea-8a63-8ba47fa178a8.png) |   ![image](https://user-images.githubusercontent.com/37788686/90134623-695c6e00-ddac-11ea-82c9-271948e77a6c.png) |  

A cascade of successive compositions can produce images which drift far from the original image, and lead to unrealistic images. But, AugMix use the parallel connection of augmentation operations, so it can produce a new image without veering too far from the original.


## Pseudocode of AugMix
![image](https://user-images.githubusercontent.com/37788686/90134284-e63b1800-ddab-11ea-80d9-c811d10b938f.png)
Jensen-Shannon Divergence can be understood to measure how similar the distributions of original, augmix1 and augmix2 are.

<img src="https://user-images.githubusercontent.com/37788686/100360548-6b38fd00-303c-11eb-9938-a5004670afd8.png" width="50%">

P usually represents the true distribution, actual observation data.
And Q is used as a theory, model, and approximation of P.


## Result
I trained CIFAR-100 with augmix and then measured mCE(mean Corruption Error) of CIFAR-100-C.
```
<Settings>
Epochs: 100  
k: 3  
severity: 3  
JSD reduction: ’batchmean’  
lambda: 12  
```
![image](https://user-images.githubusercontent.com/37788686/89794527-227b3800-db62-11ea-8326-18779f289c94.png)


|     | AugMix W/ JSD | AugMix W/O JSD | No AugMix |
| --- | :----: | :------------: | :-------: |
| mCE | 36.2%  |     47.2%      |  47.4%.   |

 

## Some AugMix Images
<p float='left'>
  <img src="./res/img_41.png" width=400px>
  <img src='./res/img_42.png' width=400px>
  <img src='./res/img_83.png' width=400px>
  <img src='./res/img_84.png' width=400px>
</p>

## To train
#### AugMix
`python3 main.py`  
#### AugMix without JSD loss
`python3 main.py --wo_jsd`  
#### No AugMix
`python3 main.py --no_jsd`

## To test
`python3 main.py --test --path {.ckpt file what you want to test}`

## Code Explanation
* augmix.py
```python
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
```
This class is dataset using AugMix method. AugMixDataset outputs 3 images(original+augmix1+augmix2). augmix1 and augmix2 is used to calculate JS-Divergence.
```python
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
```
k is length of serail connection and I set this 3. Severity is hyperparameter of how much corruption is used. augment_and_mix function is implementation of pseudocode. In outer loop, there is inner loop. Inner loop samples operator of augmentation randomly. After inner loop, original image and augmented image are mixed.
```python
    def jensen_shannon(self, logits_o, logits_1, logits_2):
        p_o = F.softmax(logits_o, dim=1)
        p_1 = F.softmax(logits_1, dim=1)
        p_2 = F.softmax(logits_2, dim=1)

        # kl(q.log(), p) -> KL(p, q)
        M = torch.clamp((p_o + p_1 + p_2) / 3, 1e-7, 1)  # to avoid exploding
        js = (self.kl(M.log(), p_o) + self.kl(M.log(), p_1) + self.kl(M.log(), p_2)) / 3
        return js
```      
jensen_shannon function is used to calculate JS-Divergence. logits_o,1,2 is distribution of probabilities of original image, augmix1 and augmix2. The smaller JS-Divergence, the similiar the distributions.

* main.py
```python
N = images[0].size(0)
ori_aug1_aug2 = torch.cat(images, dim=0).cuda()
targets = targets.cuda()
logits = model(ori_aug1_aug2)
logits_o, logits_1, logits_2 = torch.split(logits, N)

ori_loss = F.cross_entropy(logits_o, targets)
jsd = train_data.aug.jensen_shannon(logits_o, logits_1, logits_2)
loss = ori_loss + 12 * jsd
```                        
In main.py, insert original+augmix1+augmix2 data as input of model. ori_loss is cross entropy of original image. jsd is value of JS-Divergence w.r.t original, augmix1, augmix2. In my experience, ori_loss + 12*jsd is well. 12 is hyperparameter.   
## References
[1] [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty(ICLR'20)](https://arxiv.org/abs/1912.02781)

[2] https://github.com/google-research/augmix

# AugMix:Pytorch Implementation
Pytorch Implementation of [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781).

## What is AugMix?
A novel agumentation technique to improve robustness and uncertainty.
![image](https://user-images.githubusercontent.com/37788686/90134108-9eb48c00-ddab-11ea-8732-da1446729f32.png)

## Overview
### AugMix
![image](https://user-images.githubusercontent.com/37788686/90134412-197da700-ddac-11ea-8a63-8ba47fa178a8.png)

### Cascade of successive compositions
![image](https://user-images.githubusercontent.com/37788686/90134623-695c6e00-ddac-11ea-82c9-271948e77a6c.png) |  

Unlike a cascade of successive compositions, AugMix produces a new image without veering too far from the original.

## Pseudocode of AugMix
![image](https://user-images.githubusercontent.com/37788686/90134284-e63b1800-ddab-11ea-80d9-c811d10b938f.png)

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

## Usage
#### AugMix
`python3 main.py`  
#### AugMix without JSD loss
`python3 main.py --wo_jsd`  
#### No AugMix
`python3 main.py --no_jsd`
#### Test
`python3 main.py --test --path {ckpt}`


## References
[1] [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty(ICLR'20)](https://arxiv.org/abs/1912.02781)

[2] https://github.com/google-research/augmix

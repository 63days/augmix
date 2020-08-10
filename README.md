# AugMix:Pytorch Implementation
Pytorch Implementation of AugMix (ICLR2020)

## Result
```
<Settings>
Epochs: 100  
k: 3  
severity: 3  
JSD reduction: ’batchmean’  
lambda: 12  
```
![image](https://user-images.githubusercontent.com/37788686/89794527-227b3800-db62-11ea-8326-18779f289c94.png)


|     | AugMix | AugMix W/O JSD | No AugMix |
| --- | :----: | :------------: | :-------: |
| mCE | 36.2%  |     47.2%      |  47.4%.   |

## Some AugMix Images
<p float='left'>
  <img src="./res/img_41.png" width=400px>
  <img src='./res/img_42.png' width=400px>
  <img src='./res/img_83.png' width=400px>
  <img src='./res/img_84.png' width=400px>
</p>

## References
[1] [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty(ICLR'20)](https://arxiv.org/abs/1912.02781)

[2] https://github.com/google-research/augmix

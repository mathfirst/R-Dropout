# R-Dropout
This repo is a Pytorch implementation for R-Dropout which is proposed in the following paper.
```
@inproceedings{liang2021rdrop,
  title={R-Drop: Regularized Dropout for Neural Networks},
  author={Liang, Xiaobo* and Wu, Lijun* and Li, Juntao and Wang, Yue and Meng, Qi and Qin, Tao and Chen, Wei and Zhang, Min and Liu, Tie-Yan},
  booktitle={NeurIPS},
  year={2021}
}
```

Before you run my .ipynb file via Jupyter notebook, please take a look at the outputs when I ran the code. Then you can run the code and compare the results you get with mine. They are supposed to be similar.

I implemented R-Dropout just for fun and I did not explore this idea too much. This code is run on FashionMnist dataset for 10 epochs using MLP. My observation is that R-Dropout helps improve the model's performance to some extent.

The official code for R-Dropout is available at https://github.com/dropreg/R-Drop. The official code is large. My code is simple and straightforward, and also is consistent with the original paper. In my opinion, if you want to apply the idea of R-Dropout to your work, you can just modify my code to save time. The backbone of my code is as follows. You can move this code to your work directly.

```python
import torch
import torch.nn as nn

log_softmax = nn.LogSoftmax(dim=-1)
kl_loss_fn = nn.KLDivLoss(reduction="sum", log_target=True)
nll_loss_fn = nn.NLLLoss()
def r_dropout_loss(model, data, y, alpha=0.05): 
    # alpha is the regularization coefficient. It should be not too large.
    batch_size = data.size(0)
    data = torch.cat([data,data], dim=0)
    pred = model(data)
    log_probs = log_softmax(pred)
    log_probs1, log_probs2 = log_probs[:batch_size, :], log_probs[batch_size:, :]
    nll_loss = 0.5*( nll_loss_fn(log_probs1, y) + nll_loss_fn(log_probs2, y) ) # (nll_loss1+nll_loss_2)/2
    kl_loss = 0.5*( kl_loss_fn(log_probs1, log_probs2) + kl_loss_fn(log_probs2, log_probs1) ) # (KL(p||q)+KL(q||p))/2
    loss = nll_loss + alpha*kl_loss
    
    return loss
```

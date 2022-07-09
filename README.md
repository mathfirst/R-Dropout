# R-Dropout
This repo is my Pytorch implementation for R-Dropout which is proposed in the following paper. Before you run my .ipynb file via Jupyter notebook, please take a look at the outputs when I ran the code. Then you can run the code and compare the results you get with mine. They are supposed to be similar.

I did not explore too much and I did this just for fun. I only ran my code on FashionMnist dataset for 10 epochs using MLP. My observation is that R-Dropout helps improve the model's performance to some extent.

```
@inproceedings{liang2021rdrop,
  title={R-Drop: Regularized Dropout for Neural Networks},
  author={Liang, Xiaobo* and Wu, Lijun* and Li, Juntao and Wang, Yue and Meng, Qi and Qin, Tao and Chen, Wei and Zhang, Min and Liu, Tie-Yan},
  booktitle={NeurIPS},
  year={2021}
}
```

The official code for R-Dropout is available at https://github.com/dropreg/R-Drop. The official code is large. In my opinion, if you want to apply the idea of R-Dropout to your work, you can just modify my code to save time.

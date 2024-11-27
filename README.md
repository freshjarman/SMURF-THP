# SMURF-THP
Code for SMURF-THP: Score Matching-based UnceRtainty quantiFication for Transformer Hawkes Process, ICML 2023.

### Dependencies
* Python 3.7.
* [Anaconda](https://www.anaconda.com/) contains all the required packages.
* [PyTorch](https://pytorch.org/) version 1.8.0.

### Implementations



### Instructions
* Put the data folder inside the root folder (i.e. parent folder of `SMURF-THP`), modify the **data** entry in **.sh** files accordingly. The datasets are available [here](https://drive.google.com/drive/folders/0BwqmV0EcoUc8UklIR1BKV25YR1U?resourcekey=0-OrlU87jyc1m-dVMmY5aC4w&usp=sharing).
```
cd scripts/so
# train and evaluate the model
bash smurf_so.sh
```

```
# Debug training (# add -debug / or use VSCode UI to debug)
-data ../data/data_so/fold1/ -save_path ./checkpoints/so/ -save_name so_smurf_thp_debug_test -load_path_name ./checkpoints/so/so_smurf_thp_debug_test.pth -save_result ./results/so/so_smurf_thp_debug_test_samples -add_noise denoise
# Debug evaluation 
- just_eval # add behind the above command
```

## Metric: Calibration Score
* Obtain samples from langevin dynamics
* Get a confidence interval given a quantile(e.g 0.8)
* Calculate the ratio of true time falling in the interval(e.g 0.78)
* Calculate RMSE between the ratio and the quantile(e.g quantile: 0.1,0.2,...,0.9; ratio: 0.05,0.18,...,0.91)

### Reference

Please cite the following paper if you use this code.

```
@inproceedings{li2023smurf,
  title={SMURF-THP: Score Matching-based UnceRtainty quantiFication for Transformer Hawkes Process},
  author={Li, Zichong and Xu, Yanbo and Zuo, Simiao and Jiang, Haoming and Zhang, Chao and Zhao, Tuo and Zha, Hongyuan},
  booktitle = {{ICML}},
  year={2023}
}
```
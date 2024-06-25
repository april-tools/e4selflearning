# Self-supervised learning for personal sensing in mood disorders

This codebase was developed by [Filippo Corponi](https://github.com/FilippoCMC) and [Bryan M. Li](https://github.com/bryanlimy). It is part of the paper ''[Wearable data from students, teachers or subjects with alcohol use disorder help detect acute mood episodes via self-supervised learning](https://arxiv.org/abs/2311.04215)''. If you find this code or any of the ideas in the paper useful, please consider citing:

```bibtex
@article{corponi2023wearable,
  title={Wearable data from students, teachers or subjects with alcohol use disorder help detect acute mood episodes via self-supervised learning},
  author={Corponi, Filippo and Li, Bryan M and Anmella, Gerard and Valenzuela-Pascual, Cl{\`a}udia and Mas, Ariadna and Pacchiarotti, Isabella and Valent{\'\i}, Marc and Grande, Iria and Benabarre, Antonio and Garriga, Marina and others},
  journal={arXiv preprint arXiv:2311.04215},
  year={2023}
}
```

## Setup
Software development environment setup

- Create a new [conda](https://conda.io/en/latest/) environment with Python 3.10.
  ```bash
  conda create -n ssl python=3.10
  ```
- Activate `ssl` virtual environment
  ```bash
  conda activate ssl
  ```
- Install all dependencies and packages with `setup.sh` script.
  ```bash
  sh setup.sh
  ```


## Data Pre-processing
[data/README.md](data/README.md) details the structure of the dataset.

### On-/off-body & sleep/wake detection

As HR starts being recorded with a 10-second lag with respect to other channels, the first 10 seconds are dropped from channels other than HR. While channels should all stop at the same time, as a failsafe, channels are cropped to the shortest channel duration. 

#### On-/off-body detection

We considered measurements smaller than 0.05 μS as indicative of off-body status. Furthermore, as we noticed occurrences of values greater than the EDA sensor range (i.e., 100 μS), as well as instances of TEMP values outside the physiological range (30-40°C), we set both to off-body. On-body sequences need to last more than a given number of minutes (specified with `--wear_minimum_minutes`) otherwise they are set to off-body.

#### Sleep/wake detection

`--sleep_algorithm` specifies which one of [van Hees et al. 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0142533) and [Scripps](https://onlinelibrary.wiley.com/doi/10.1111/j.1365-2869.2010.00835.x) algorithms to use. Note that these require a minimum of on-body recording time to operate. A mask is returned where wake = 0, sleep = 1, off-body = 2. 

```bash
python preprocess_ds.py --output_dir data/preprocessed/unsegmented --overwrite
```

Please see `--help` for all available options. `data/preprocessed/unsegmented` contains preprocessed recordings. Specifically, each folder maps to a preprocessed recoding. `channels.h5` is a dictionary storing processed E4 channels, i.e.  `ACC`,  `BVP`,  `EDA`,  `HR`,  `IBI`,  `TEMP` as well as the masks computed during the preprocessing.

---

### Segmentation
 Segmentation is carried out on sleep/wake sequences independently. Sleep/wake status for a given segment is saved as part of that segment label. Segmentation returns recording segments, whose length is set with `--segment_length`, containg the following channels: `ACC_x`, `ACC_y`, `ACC_z`, `BVP`, `EDA`, `TEMP`. `HR` and `IBI`, since they are both derived from `BVP`, are not used in deep-learning models. The optional flag `--flirt` adds an extra channel to the segments named `FLIRT`, which contains acc, eda, and hrv features extracted on the segment, with feature generation toolkit [FLIRT](https://flirt.readthedocs.io/en/latest/index.html). As `FLIRT` does not provide any built-in feature extractor for temperature, we extracted the average and standard deviation across the segment. Only a single row of features is derived per segment (in other words, window_length for FLIRT is set equal to segment_length). FLIRT features are only computed on labelled segments, that is segments from recordings collected and annotated at Hospital Clìnic, Barcelona.

  ```bash
  python segment.py --output_dir data/preprocessed/sl512_ss128 --segmentation_mode 1 --segment_length 512 --step_size 128 --overwrite
  ```
  Please see `--help` for all available options.

## Exacerbation vs Euthymia detection

The target task is time-series (binary) classification, specifically identifying whether a recording segment was taken from a subject experiencing an acute mood disorder episode of any polarity (depression, mania, mixed features) or from someone with an historical mood disorder diagnosis but clinically stable at the time of recording (a condition referred to as [euthymia](https://www.sciencedirect.com/book/9780124051706/clinical-trial-design-challenges-in-mood-disorders) in psychiatric parlance).

---

### Fully-supervised learning

#### Classical Machine Learning (XGboost)

```bash
python train_cml.py --dataset data/preprocessed/sl512_ss128 --output_dir runs/sl_xgboost_test --clear_output_dir
```

#### Deep-learning

```bash
python train_ann.py --task_mode 3 --dataset data/preprocessed/sl512_ss128 runs/sl_ann_test
```

---

### Self-supervised learning

#### Pre-training on masked prediction

```bash
python pre_train.py --pretext_task masked_prediction --dataset data/preprocessed/sl512_ss128 --output_dir runs/masked_prediction_test
```
The `--unlabelled_data_resampling_percentage` and `--filter_collections` flags are used for ablations analyses.

#### Fine-tuning on the main task

```bash
python train_ann.py --task_mode 1 --path2pretraining_res runs/masked_prediction_test --output_dir runs/masked_prediction_fine_tuning_test
```

Please see `--help` for all available options. 

## Open-access datasets recording with an Empatica E4

We herewith acknowledge and list the open-access datasets recording with an [Emaptica E4](https://support.empatica.com/hc/en-us/articles/202581999-E4-wristband-technical-specifications) which were used for self-supervised pre-training:
- [ADARP](https://arxiv.org/abs/2206.14568) by Sah et al. 2022
- [BID IDEAS Lab](https://www.nature.com/articles/s41746-021-00465-w) by Bent et al. 2021
- [In-Gauge En-Gage](https://www.nature.com/articles/s41597-022-01347-w) by Gao et al. 2022
- [Nurses Stress Detection](https://www.nature.com/articles/s41597-022-01361-y) by Hosseini et al. 2022
- [PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia) by Reiss et al. 2019
- [SPS](https://www.mdpi.com/1424-8220/22/21/8135) by Iqbal et al. 2022
- [Toadstool](https://dl.acm.org/doi/10.1145/3339825.3394939) by Svoren et al. 2020
- [UE4W](https://zenodo.org/record/6898244) by Hinkle et al. 2022
- [WEEE](https://www.nature.com/articles/s41597-022-01643-5) by Gashi et al. 2022
- [WESAD](https://dl.acm.org/doi/10.1145/3242969.3242985) by Schmidt et al. 2018
- [WESD](https://ieeexplore.ieee.org/abstract/document/9744065) by Amin et al. 2022

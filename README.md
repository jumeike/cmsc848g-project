# CMSC848G: Deep Learning with PyTorch

> CIFAR-10 image classification experiments — Part 1: minimal CNN adaptation, Part 2: VGG11 ablation study.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jumeike/cmsc848g-project/blob/main/notebook.ipynb)

---

## Overview

| Part | Description | Result |
|------|-------------|--------|
| Part 1 | Adapt PyTorch MNIST CNN to CIFAR-10 (2 changes) | 73% |
| Part 2 | VGG11 ablation: BN, activation, optimizer, init, dropout | 62 - 81% |

---

## Project Structure

```
scripts/
  cifar10_cnn.py      Part 1 training script
  vgg_cifar10.py      Part 2 training script (all experiments)
  plot_part1.py       Plot Part 1 results
  plot_part2.py       Plot Part 2 results
results/
  part1/              Part 1 log
  part2/              Part 2 experiment logs (7 runs)
notebook.ipynb        Interactive Colab notebook (both parts)
run.sh                GPU job launcher (Nexus/SLURM)
```

---

## Setup

```bash
conda activate cmsc848g
```

---

## Running

**Part 1**
```bash
python scripts/cifar10_cnn.py | tee results/part1/part1_output.log
```

**Part 2** (one experiment at a time, e.g. 1a)
```bash
python scripts/vgg_cifar10.py --model vgg11 --activation relu \
    | tee results/part2/1a_vgg11_relu.log
```

All Part 2 options: `python scripts/vgg_cifar10.py --help`

---

## Plotting

```bash
python scripts/plot_part1.py    # -> figures/part1_plot.png
python scripts/plot_part2.py    # -> figures/part2_section1.png, part2_section2.png, part2_comparison.png
```

---

## GPU Allocation (Nexus cluster)

```bash
./run.sh                          # default: 1x RTX A6000, 8h
./run.sh -t 2:00:00 -g rtx3090   # custom options
```

> **No cluster access?** Open `notebook.ipynb` in [Google Colab](https://colab.research.google.com) with a free T4 GPU. Experiments run at 25 epochs (full runs used 100 epochs on an A6000).

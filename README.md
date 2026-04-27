# CMSC848G Project: Deep Learning with PyTorch

CIFAR-10 image classification experiments using a minimal CNN and VGG11.

## Structure

```
scripts/          Training and plotting scripts
results/          Experiment logs (part1, part2)
figures/          Generated plots
run.sh            Interactive GPU job launcher (Nexus/SLURM)
```

## Setup

```bash
conda activate cmsc848g
```

## Running Experiments

**Part 1** — Adapted MNIST CNN on CIFAR-10:
```bash
python scripts/cifar10_cnn.py | tee results/part1/part1_output.log
```

**Part 2** — VGG11 experiments (run from project root):
```bash
# Example: experiment 1a
python scripts/vgg_cifar10.py --model vgg11 --activation relu \
    | tee results/part2/1a_vgg11_relu.log
```

See `scripts/vgg_cifar10.py --help` for all options.

## Plotting

```bash
python scripts/plot_part1.py   # figures/part1_plot.png
python scripts/plot_part2.py   # figures/part2_section1.png, part2_section2.png, part2_comparison.png
```

## GPU Allocation (Nexus cluster)

```bash
./run.sh                        # default: 1x RTX A6000, 8h
./run.sh -t 2:00:00 -g rtx3090
```

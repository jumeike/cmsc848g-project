import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = 'results/part2'
FIGURES_DIR = 'figures'

EXPERIMENTS = [
    ('1a_vgg11_relu.log',       '1a: VGG11, ReLU',          76),
    ('1b_vgg11_bn_relu.log',    '1b: VGG11-BN, ReLU',       82),
    ('1c_vgg11_leaky_relu.log', '1c: VGG11, LeakyReLU',     75),
    ('2a_sgd.log',              '2a: LeakyReLU + SGD',       62),
    ('2b_bs256.log',            '2b: LeakyReLU + BS=256',    75),
    ('2c_xavier.log',           '2c: LeakyReLU + Xavier',    81),
    ('2d_no_dropout.log',       '2d: LeakyReLU, No Dropout', 73),
]

REFERENCES = {
    '1a': 78, '1b': 83, '1c': 74,
    '2a': 66, '2b': 77, '2c': 82, '2d': 67,
}

FONT = 11


def parse_log(path):
    epochs, losses, accs = [], [], []
    with open(path) as f:
        for line in f:
            m = re.match(r'Epoch (\d+)\s+Loss: ([\d.]+)\s+Accuracy: ([\d.]+)', line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                accs.append(float(m.group(3)))
    return epochs, losses, accs


# ── Figure 1: Section 1 — Model components ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 8))

section1 = EXPERIMENTS[:3]
for i, (fname, label, _) in enumerate(section1):
    epochs, losses, accs = parse_log(f'{RESULTS_DIR}/{fname}')
    ref = REFERENCES[fname[:2]]
    axes[0, i].plot(epochs, losses, marker='o', markersize=2)
    axes[0, i].set_title(label, fontsize=FONT + 1, fontweight='bold')
    axes[0, i].set_xlabel('Epoch', fontsize=FONT)
    axes[0, i].set_ylabel('Train Loss', fontsize=FONT)
    axes[0, i].tick_params(labelsize=FONT - 1)
    axes[0, i].grid(True, alpha=0.4)

    axes[1, i].plot(epochs, accs, marker='o', markersize=2, color='orange')
    axes[1, i].axhline(ref, color='red', linestyle='--', linewidth=1, label=f'Ref: {ref}%')
    axes[1, i].set_title(label, fontsize=FONT + 1, fontweight='bold')
    axes[1, i].set_xlabel('Epoch', fontsize=FONT)
    axes[1, i].set_ylabel('Test Accuracy (%)', fontsize=FONT)
    axes[1, i].tick_params(labelsize=FONT - 1)
    axes[1, i].legend(fontsize=FONT - 1)
    axes[1, i].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/part2_section1.png', dpi=150)
print('Saved figures/part2_section1.png')
plt.close()

# ── Figure 2: Section 2 — Training methods ──────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(22, 8))

section2 = EXPERIMENTS[3:]
for i, (fname, label, _) in enumerate(section2):
    epochs, losses, accs = parse_log(f'{RESULTS_DIR}/{fname}')
    ref = REFERENCES[fname[:2]]
    axes[0, i].plot(epochs, losses, marker='o', markersize=2)
    axes[0, i].set_title(label, fontsize=FONT, fontweight='bold')
    axes[0, i].set_xlabel('Epoch', fontsize=FONT)
    axes[0, i].set_ylabel('Train Loss', fontsize=FONT)
    axes[0, i].tick_params(labelsize=FONT - 1)
    axes[0, i].grid(True, alpha=0.4)

    axes[1, i].plot(epochs, accs, marker='o', markersize=2, color='orange')
    axes[1, i].axhline(ref, color='red', linestyle='--', linewidth=1, label=f'Ref: {ref}%')
    axes[1, i].set_title(label, fontsize=FONT, fontweight='bold')
    axes[1, i].set_xlabel('Epoch', fontsize=FONT)
    axes[1, i].set_ylabel('Test Accuracy (%)', fontsize=FONT)
    axes[1, i].tick_params(labelsize=FONT - 1)
    axes[1, i].legend(fontsize=FONT - 1)
    axes[1, i].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/part2_section2.png', dpi=150)
print('Saved figures/part2_section2.png')
plt.close()

# ── Figure 3: Final accuracy comparison bar chart ───────────────────────────
labels = [e[1] for e in EXPERIMENTS]
results = [parse_log(f'{RESULTS_DIR}/{e[0]}')[2][-1] for e in EXPERIMENTS]
refs = [REFERENCES[e[0][:2]] for e in EXPERIMENTS]

x = range(len(labels))
fig, ax = plt.subplots(figsize=(16, 5))

bars1 = ax.bar(
    [i - 0.22 for i in x], results, width=0.42,
    label='Result', color='steelblue', edgecolor='black', linewidth=0.8,
)
bars2 = ax.bar(
    [i + 0.22 for i in x], refs, width=0.42,
    label='Reference', color='#E07B54', edgecolor='black', linewidth=0.8,
)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=FONT - 1)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=FONT - 1)

ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=12, ha='right', fontsize=FONT)
ax.set_ylabel('Test Accuracy (%)', fontsize=FONT + 1)
ax.tick_params(axis='y', labelsize=FONT)
ax.set_ylim(50, 97)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
ax.legend(fontsize=FONT, framealpha=0.9)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/part2_comparison.png', dpi=150)
print('Saved figures/part2_comparison.png')
plt.close()

import re
import matplotlib.pyplot as plt

log_file = 'results/part1/part1_output.log'

train_losses = {}  # epoch -> list of losses
test_accuracies = {}  # epoch -> accuracy

with open(log_file) as f:
    for line in f:
        m = re.match(r'Train Epoch: (\d+) .*\s+Loss: ([\d.]+)', line)
        if m:
            epoch, loss = int(m.group(1)), float(m.group(2))
            train_losses.setdefault(epoch, []).append(loss)
            continue
        m = re.match(r'Test set: .* Accuracy: \d+/\d+ \(([\d.]+)%\)', line)
        if m:
            # associate with the last seen epoch
            acc = float(m.group(1))
            last_epoch = max(train_losses.keys()) if train_losses else 0
            test_accuracies[last_epoch] = acc

epochs = sorted(train_losses.keys())
avg_losses = [sum(train_losses[e]) / len(train_losses[e]) for e in epochs]
accuracies = [test_accuracies.get(e, None) for e in epochs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(epochs, avg_losses, marker='o')
ax1.set_title('Train Loss per Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Avg Loss')
ax1.grid(True)

ax2.plot(epochs, accuracies, marker='o', color='orange')
ax2.set_title('Test Accuracy per Epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True)

plt.tight_layout()
plt.savefig('figures/part1_plot.png', dpi=150)
print('Saved figures/part1_plot.png')

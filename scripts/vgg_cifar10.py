import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models


def replace_relu(model):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, nn.LeakyReLU(inplace=True))
        else:
            replace_relu(module)


def disable_dropout(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Dropout):
            setattr(model, name, nn.Identity())
        else:
            disable_dropout(module)


def init_weights(model, init_type):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def build_model(args):
    if args.model == 'vgg11_bn':
        model = models.vgg11_bn(weights=None)
    else:
        model = models.vgg11(weights=None)

    # Adapt for CIFAR10: replace final classifier layer (1000 -> 10)
    model.classifier[6] = nn.Linear(4096, 10)

    if args.activation == 'leaky_relu':
        replace_relu(model)

    if args.no_dropout:
        disable_dropout(model)

    init_weights(model, args.init)

    return model


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return total_loss / len(train_loader)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return accuracy


def get_exp_name(args):
    parts = [
        args.model,
        args.activation,
        f'opt_{args.optimizer}',
        f'bs{args.batch_size}',
        f'init_{args.init}',
    ]
    if args.no_dropout:
        parts.append('no_dropout')
    return '_'.join(parts)


def main():
    parser = argparse.ArgumentParser(description='VGG on CIFAR10')
    parser.add_argument('--model', default='vgg11', choices=['vgg11', 'vgg11_bn'],
                        help='model variant (default: vgg11)')
    parser.add_argument('--activation', default='relu', choices=['relu', 'leaky_relu'],
                        help='activation function (default: relu)')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'],
                        help='optimizer (default: adam)')
    parser.add_argument('--init', default='kaiming', choices=['kaiming', 'xavier'],
                        help='Conv2d weight init (default: kaiming)')
    parser.add_argument('--no-dropout', action='store_true',
                        help='disable dropout layers')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='training batch size (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='test batch size (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 2e-4)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='batches between log prints (default: 50)')
    parser.add_argument('--save-model', action='store_true',
                        help='save model after training')
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()
    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device('cpu')

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 2, 'persistent_workers': True,
                        'pin_memory': True, 'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset_train = datasets.CIFAR10('../data', train=True, download=True,
                                     transform=transform_train)
    dataset_test = datasets.CIFAR10('../data', train=False,
                                    transform=transform_test)
    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = build_model(args).to(device)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    exp_name = get_exp_name(args)
    os.makedirs('logs', exist_ok=True)
    log_path = f'logs/{exp_name}.log'

    print(f'Experiment: {exp_name}')
    print(f'Device: {device}')
    print(f'Log: {log_path}\n')

    with open(log_path, 'w') as f:
        f.write(f'Experiment: {exp_name}\n')

    for epoch in range(1, args.epochs + 1):
        avg_loss = train(args, model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        with open(log_path, 'a') as f:
            f.write(f'Epoch {epoch}\tLoss: {avg_loss:.6f}\tAccuracy: {accuracy:.2f}\n')

    if args.save_model:
        torch.save(model.state_dict(), f'{exp_name}.pt')


if __name__ == '__main__':
    main()

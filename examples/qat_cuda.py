import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from tqdm import tqdm
from prettytable import PrettyTable
from pathlib import Path
import pycomposit as c

from gpuds import train_loader, test_loader, image_transform, dsetname, dsetnclasses

########### REFERENCE QAT !#################

torch.set_float32_matmul_precision("high")

runtag = f"composit-{dsetname}"

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, dsetnclasses)  # CIFAR-10 has 10 classes

# Training setup
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


@torch.compile
def train_epoch(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        inputs = image_transform(inputs)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / len(trainloader), 100.0 * correct / total


@torch.compile
def test_epoch(model, testloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return test_loss / len(testloader), 100.0 * correct / total


test_loss, test_acc = test_epoch(model.to(device), test_loader, criterion, device)
print(f"Float32 Initial Test Loss: {test_loss:.4f}, Test Acc: {test_acc:6.2f}%")

# # Training loop
num_epochs = 2
best_acc = 0
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test_epoch(model, test_loader, criterion, device)

    scheduler.step()

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "best_float_model.pth")

    if epoch % 1 == 0 or epoch == num_epochs - 1:
        print(
            f"Float Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:6.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:6.2f}%"
        )

# print(f'Best test accuracy: {best_acc:.2f}%')

########### QAT ###############

scale = 1 / 32.04
quantized_model = c.models.quantize_model(model, scale)
quantized_model = quantized_model.to(device)

print(quantized_model)

# Training setup
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(quantized_model.parameters(), lr=1e-4, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

test_loss, test_acc = test_epoch(quantized_model.to(device), test_loader, criterion, device)
print(f"Quant Initial Test Loss: {test_loss:.4f}, Test Acc: {test_acc:6.2f}%")

# # Training loop
num_epochs = 5000
best_acc = 0
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(quantized_model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test_epoch(quantized_model, test_loader, criterion, device)

    scheduler.step()

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(quantized_model.state_dict(), "best_quant_model.pth")

    if epoch % 1 == 0 or epoch == num_epochs - 1:
        print(
            f"Quant Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:6.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:6.2f}%"
        )


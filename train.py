import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import PrunableNet
from utils import sparsity_loss, calculate_sparsity, evaluate, plot_gates

# 🔹 Hyperparameters
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 10
LAMBDAS = [0, 0.05, 0.1, 0.2]   # Step 8

# 🔹 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), 
                         (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)

results = []
# 🔹 Run experiments for different λ
for LAMBDA in LAMBDAS:
    print(f"\n🚀 Training with Lambda = {LAMBDA}\n")
    
    model = PrunableNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 🔹 Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        ce_total = 0
        sp_total = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            ce_loss = criterion(outputs, labels)
            sp_loss = sparsity_loss(model)
            
            loss = ce_loss + LAMBDA * sp_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            ce_total += ce_loss.item()
            sp_total += sp_loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        print(f"Epoch {epoch+1}, CE: {ce_total:.4f}, SP: {sp_total:.4f}")
    
    for g in model.get_all_gates():
        print(g.min().item(), g.max().item(), g.mean().item())
    # 🔹 Evaluation
    accuracy = evaluate(model, testloader, device)
    sparsity = calculate_sparsity(model)
    if LAMBDA == 0.05:
        plt.figure()
        plot_gates(model)
        plt.savefig("gate_distribution.png")

    print(f"\nLambda: {LAMBDA}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Sparsity: {sparsity:.2f}%")
    results.append((LAMBDA, accuracy, sparsity))   # 👈 ADD THIS
print("\n📊 Final Results:")
for l, a, s in results:
    print(f"Lambda: {l} | Accuracy: {a:.2f}% | Sparsity: {s:.2f}%")
    
# 🔹 Plot for last model

lambdas, accs, sparsities = zip(*results)

plt.figure()
plt.plot(sparsities, accs, marker='o')
plt.xlabel("Sparsity (%)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Sparsity Tradeoff")
plt.grid()

# 🔥 SAVE GRAPH
plt.savefig("accuracy_vs_sparsity.png")
plt.show()
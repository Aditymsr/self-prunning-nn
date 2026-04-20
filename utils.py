import torch
import matplotlib.pyplot as plt

# 🔹 Sparsity Loss
def sparsity_loss(model):
    loss = 0
    for gates in model.get_all_gates():
        loss += (gates * (1 - gates)).mean()
    return loss


# 🔹 Sparsity Calculation
def calculate_sparsity(model, threshold=0.3):
    total = 0
    zero = 0
    
    for gates in model.get_all_gates():
        total += gates.numel()
        zero += (gates < threshold).sum().item()
    
    return 100 * zero / total


# 🔹 Evaluation
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total


# 🔹 Plot Gate Distribution
def plot_gates(model):
    all_gates = []
    
    for g in model.get_all_gates():
        all_gates.extend(g.detach().cpu().numpy().flatten())
    
    plt.hist(all_gates, bins=50)
    plt.title("Gate Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    #plt.show()

"""
@author: Akash Maji
@email: akashmaji@iisc.ac.in
@intent: implementing lenet5 using pytorch
@reference: https://pytorch.org/docs/stable/nn.html#
"""

# import necessary libraries
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torchvision  
import torchvision.transforms as transforms  
import matplotlib.pyplot as plt  

# Set device  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# Define depth-wise separable convolution function  
def depthwise_separable_conv(in_channels, out_channels, kernel_size, stride=1, padding=0):  
    return nn.Sequential(  
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels),  # Depthwise  
        nn.Conv2d(in_channels, out_channels, kernel_size=1)  # Pointwise  
    )  

# Load MNIST dataset  
transform = transforms.Compose([  
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]  
])  

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)  
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)  

training_loaded_data = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)  
test_loaded_data = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)  

# Define layers with depth-wise separable convolutions  
convolution_layer_1 = depthwise_separable_conv(1, 6, kernel_size=5, padding=2).to(device)  # Input: 28x28, Output: 24x24  
pooling_1 = nn.AvgPool2d(kernel_size=2, stride=2).to(device)  # Average pooling  

convolution_layer_2 = depthwise_separable_conv(6, 16, kernel_size=5).to(device)  # Input: 12x12, Output: 8x8  
pooling_2 = nn.AvgPool2d(kernel_size=2, stride=2).to(device)  # Average pooling  

fully_connected_layer_1 = nn.Linear(16 * 5 * 5, 120).to(device)  
fully_connected_layer_2 = nn.Linear(120, 84).to(device)  
fully_connected_layer_3 = nn.Linear(84, 10).to(device)  

# Define loss and optimizer  
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(list(convolution_layer_1.parameters()) + list(convolution_layer_2.parameters()) +  
                       list(fully_connected_layer_1.parameters()) + list(fully_connected_layer_2.parameters()), lr=0.001)  

# Training loop  
num_epochs = 20  
training_loss, training_accuracy = [], []  

for epoch in range(num_epochs):  
    running_loss, correct, total = 0.0, 0, 0  
    for inputs, labels in training_loaded_data:  
        inputs, labels = inputs.to(device), labels.to(device)  

        # Forward pass  
        x = pooling_1(torch.sigmoid(convolution_layer_1(inputs)))  # First layer  
        x = pooling_2(torch.sigmoid(convolution_layer_2(x)))       # Second layer  
        x = x.view(-1, 16 * 5 * 5)                                # Flatten  
        x = torch.sigmoid(fully_connected_layer_1(x))             # Fully connected layer 1  
        x = torch.sigmoid(fully_connected_layer_2(x))             # Fully connected layer 2  
        outputs = fully_connected_layer_3(x)                      # Output layer  

        # Compute loss  
        loss = criterion(outputs, labels)  
        optimizer.zero_grad()                                     # Zero gradients  
        loss.backward()                                           # Backpropagation  
        optimizer.step()                                          # Update parameters  

        # Collect metrics  
        running_loss += loss.item()  
        _, predicted = torch.max(outputs.data, 1)  
        total += labels.size(0)  
        correct += (predicted == labels).sum().item()  

    # Store epoch metrics  
    training_loss.append(running_loss / len(training_loaded_data))  
    training_accuracy.append(100 * correct / total)  
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {training_loss[-1]:.4f}, Accuracy: {training_accuracy[-1]:.2f}%")  

# Plot accuracy and loss  
plt.figure(figsize=(12, 5))  

plt.subplot(1, 2, 1)  
plt.plot(range(num_epochs), training_loss, label="Training Loss (Depth-wise Separable)", marker='o', linestyle='-', color='b')  
plt.xlabel("Epochs")  
plt.ylabel("Loss")  
plt.legend()  

plt.subplot(1, 2, 2)  
plt.plot(range(num_epochs), training_accuracy, label="Training Accuracy (Depth-wise Separable)", marker='o', linestyle='-', color='r')  
plt.xlabel("Epochs")  
plt.ylabel("Accuracy")  
plt.legend()  
plt.show()
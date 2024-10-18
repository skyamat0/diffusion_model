import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from denoising_score import ScoreNet, dsm

#訓練データ
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download = True)
#検証データ
test_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=False,
                                           transform=transforms.ToTensor(),
                                           download = True)
batch_size = 600
train_loader = torch.utils.data.DataLoader(
    train_dataset,         
    batch_size=batch_size,  
    shuffle=True,          
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
epochs = 1000
sigma = 0.1
score = ScoreNet(784)
optimizer = torch.optim.Adam(score.parameters(), lr=1e-3)
for epoch in range(epochs):
    for x, _ in train_loader:
        x = x.view(x.size(0), -1).to(device)
        x_tilde = x + torch.randn_like(x) * sigma
        target = -(x_tilde - x)/(sigma**2)
        s_theta = score(x)
        loss = dsm(s_theta, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if epoch == 0 or (epoch+1) %100 == 0:
        print(f"epoch={epoch+1}, loss={loss.item()}")
torch.save(score, 'model.pth')
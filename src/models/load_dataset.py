
import torch
import torchvision
import torchvision.transforms as transforms


def download_dataset(batch_size=128):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233 ,0.24348505 ,0.26158768))])

    batch_size = 128
    print("Downloading dataset -----")
    trainset = torchvision.datasets.CIFAR10(root='../../data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../../data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    print("Finish Downloading dataset -----")
    
    return trainloader,testloader

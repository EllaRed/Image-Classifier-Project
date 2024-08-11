import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import os

def get_input_args():
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset.")
    
    # Basic command-line arguments
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the checkpoint')
    parser.add_argument('--arch', type=str, default='vgg13', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    return parser.parse_args()

def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }
    
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    }
    
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
    }
    
    return dataloaders, image_datasets['train'].class_to_idx

def build_model(arch='vgg13', hidden_units=512, learning_rate=0.001):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError(f"Architecture {arch} not recognized.")
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer

def train_model(model, criterion, optimizer, dataloaders, device, epochs):
    model.to(device)
    
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation step
        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                
                logps = model(inputs)
                valid_loss += criterion(logps, labels).item()
                
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        model.train()
        
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. "
              f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
              f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
    
    return model

def save_checkpoint(model, save_dir, class_to_idx, arch, hidden_units, learning_rate, epochs):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx
    }
    
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))

def main():
    args = get_input_args()
    
    dataloaders, class_to_idx = load_data(args.data_dir)
    model, criterion, optimizer = build_model(arch=args.arch,
                                              hidden_units=args.hidden_units,
                                              learning_rate=args.learning_rate)
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = train_model(model, criterion, optimizer, dataloaders, device, args.epochs)
    
    save_checkpoint(model, args.save_dir, class_to_idx, args.arch, args.hidden_units, args.learning_rate, args.epochs)

if __name__ == "__main__":
    main()

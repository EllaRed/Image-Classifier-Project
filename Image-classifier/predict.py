import argparse
import torch
from torchvision import models
from PIL import Image
import numpy as np
from torch import nn
import json

def get_input_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image with a trained model.")
    
    # Basic command-line arguments
    parser.add_argument('image_path', type=str, help='Path to image')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default=None, help='Path to JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    return parser.parse_args()

def load_checkpoint(filepath):
    if(torch.cuda.is_available()):
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    if 'arch' not in checkpoint:
        checkpoint['arch'] = 'vgg16'
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    
    return model

def process_image(image_path):
    image = Image.open(image_path)
    
    # Resize and crop
    if image.size[0] > image.size[1]:
        image.thumbnail((image.size[0], 256))
    else:
        image.thumbnail((256, image.size[1]))
    
    left_margin = (image.width - 224) / 2
    bottom_margin = (image.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    image = image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Normalize
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose the color channel
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, model, top_k=5, device='cpu'):
    # Process the image
    image = process_image(image_path)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Set model to evaluation mode and disable gradient computation
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model.forward(image_tensor)
    
    # Get top K probabilities and indices
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top_probs, top_indices = probabilities.topk(top_k)
    top_probs = top_probs.cpu().numpy().squeeze()
    top_indices = top_indices.cpu().numpy().squeeze()
    
    # Convert indices to actual class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]
    
    return top_probs, top_classes

def main():
    args = get_input_args()
    
    model = load_checkpoint(args.checkpoint)
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    top_probs, top_classes = predict(args.image_path, model, top_k=args.top_k, device=device)
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_classes = [cat_to_name[cls] for cls in top_classes]
    
    print("Top probabilities:", top_probs)
    print("Top classes:", top_classes)

if __name__ == "__main__":
    main()



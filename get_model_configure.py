import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time
import os
import csv
import re

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def load_cifar10_data(data_folder, model_folder):
    if model_folder == 'vit_b_16':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])

    test_set = datasets.CIFAR10(root=data_folder, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
    return test_loader

def test_model_inference_and_accuracy(model, device, data_loader):
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    end_time = time.time()
    accuracy = 100 * correct / total
    return end_time - start_time, accuracy

def write_result_to_csv(model_folder, variant, inference_time, accuracy, models_dir):
    filename = os.path.join(models_dir, model_folder, f"{model_folder}_inference_results.csv")
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['Model', 'Variant', 'Pruning Amount', 'Inference Time (s)', 'Accuracy (%)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        # Extract pruning amount or check if it's the original model
        pruning_match = re.search(r'pruned_(\d+\.\d+)', variant)
        pruning_amount = pruning_match.group(1) if pruning_match else 'original'
        
        writer.writerow({
            'Model': model_folder, 
            'Variant': variant, 
            'Pruning Amount': pruning_amount, 
            'Inference Time (s)': f"{inference_time:.3f}", 
            'Accuracy (%)': accuracy
        })

def scan_models(models_dir):
    model_details = {}
    for folder in os.listdir(models_dir):
        folder_path = os.path.join(models_dir, folder)
        if os.path.isdir(folder_path):
            model_details[folder] = [file for file in os.listdir(folder_path) if file.endswith('.pth')]
    return model_details

def main(models_dir, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_details = scan_models(models_dir)

    for folder, files in model_details.items():
        test_data_loader = load_cifar10_data(data_dir, folder)
        for file_name in files:
            model_path = os.path.join(models_dir, folder, file_name)
            model = load_model(model_path, device)
            inference_time, accuracy = test_model_inference_and_accuracy(model, device, test_data_loader)
            write_result_to_csv(folder, file_name, inference_time, accuracy, models_dir)

if __name__ == "__main__":
    models_directory = "./models"
    data_directory = "./data"
    main(models_directory, data_directory)

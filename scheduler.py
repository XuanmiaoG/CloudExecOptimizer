import csv
import os
import time
import torch
import heapq
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class Task:
    def __init__(self, model_type, dataset, batch_size, start_time, deadline):
        self.model_type = model_type
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.start_time = int(start_time)
        self.deadline = int(deadline)
        self.priority = None  # Initially, priority is not set
        self.missed_deadline = False

    def __lt__(self, other):
        # Tasks are compared based on start_time for scheduling
        return self.start_time < other.start_time

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def load_data_loader(dataset_name, data_directory, batch_size, model_type):
    # Handle specific normalization for ViT
    if model_type == 'vit_b_16':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
    dataset = datasets.CIFAR10(root=data_directory, train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

def select_best_model_variant(model_type, models_dir, deadline, batch_size):
    csv_file_path = os.path.join(models_dir, model_type, f"{model_type}_inference_results.csv")
    models = []
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            inference_time = float(row['Inference Time (s)']) * (batch_size / 100)
            accuracy = float(row['Accuracy (%)'])
            priority = max(0, inference_time - deadline) - accuracy
            models.append({'variant': row['Variant'], 'inference_time': inference_time, 'accuracy': accuracy, 'priority': priority})
    return evaluate_pareto_optimality(models)

def evaluate_pareto_optimality(models):
    pareto_optimal = []
    for current in models:
        dominated = False
        for other in pareto_optimal:
            if dominates(other, current):
                dominated = True
                break
        if not dominated:
            pareto_optimal = [other for other in pareto_optimal if not dominates(current, other)]
            pareto_optimal.append(current)
    return min(pareto_optimal, key=lambda x: x['priority']) if pareto_optimal else None

def dominates(a, b):
    return a['inference_time'] <= b['inference_time'] and a['accuracy'] >= b['accuracy']

def execute_task(task, models_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = load_data_loader(task.dataset, './data', task.batch_size, task.model_type)
    best_model = select_best_model_variant(task.model_type, models_dir, task.deadline, task.batch_size)
    if best_model:
        model_path = os.path.join(models_dir, task.model_type, best_model['variant'])
        model = load_model(model_path, device)
        model.to(device)  # Ensure model is on the correct device
        start_time = time.time()

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)  # Move images to the same device as the model
                outputs = model(images)
        
        elapsed_time = time.time() - start_time
        elapsed_time_ms = elapsed_time * 1000  # Convert to milliseconds

        # Check if the task met its deadline
        if elapsed_time_ms > task.deadline:
            task.missed_deadline = True
            print(f"Task for {task.model_type} missed the deadline, taking {elapsed_time_ms:.2f}ms")

        print(f"Task for {task.model_type} completed in {elapsed_time:.2f}s with model variant {best_model['variant']}")
    else:
        print(f"No suitable model variant found for {task.model_type}")

def read_task_definitions(csv_file_path):
    tasks = []
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            task = Task(row['model_type'], row['dataset'], row['batch_size'], row['start_time_ms'], row['deadline_ms'])
            heapq.heappush(tasks, task)
    return tasks

def main(csv_file_path, models_dir):
    tasks = read_task_definitions(csv_file_path)
    results = []
    while tasks:
        next_task = heapq.heappop(tasks)
        execute_task(next_task, models_dir)
        results.append(next_task)

    # Calculate deadline miss rate
    total_tasks = len(results)
    missed_count = sum(1 for task in results if task.missed_deadline)
    deadline_miss_rate = (missed_count / total_tasks) * 100 if total_tasks > 0 else 0
    print(f"Total tasks: {total_tasks}")
    print(f"Tasks that met the deadline: {total_tasks - missed_count}")
    print(f"Tasks that missed the deadline: {missed_count}")
    print(f"Deadline Miss Rate: {deadline_miss_rate:.2f}%")

if __name__ == "__main__":
    main('./task_definitions.csv', './models')

import csv
import os
import time
import torch
import heapq
import random
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from deap import base, creator, tools, algorithms

# Define the problem object using DEAP's creator
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # Minimize miss chance, maximize accuracy
creator.create("Individual", list, fitness=creator.FitnessMulti)

class Task:
    def __init__(self, model_type, dataset, batch_size, start_time, deadline):
        self.model_type = model_type
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.start_time = int(start_time)
        self.deadline = int(deadline)
        self.priority = float('inf')
        self.variant = None

    def __lt__(self, other):
        return self.priority < other.priority

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def load_data_loader(dataset_name, data_directory, batch_size, model_type):
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

def predict_ddl_miss_chance(task, models_dir):
    models = []
    csv_file_path = os.path.join(models_dir, task.model_type, f"{task.model_type}_inference_results.csv")
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            inference_time = float(row['Inference Time (s)']) * (task.batch_size / 100)
            accuracy = float(row['Accuracy (%)'])
            miss_chance = max(0, inference_time - (task.deadline / 1000))  # Convert deadline to seconds
            models.append({'variant': row['Variant'], 'miss_chance': miss_chance, 'accuracy': accuracy})
    return models

def setup_toolbox(models, toolbox):
    max_index = len(models) - 1

    def create_gene():
        return random.uniform(0, 1)  # Create a single float between 0 and 1

    toolbox.register("attr_float", create_gene)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)  # List with one float
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: evaluate_individual(ind, models))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

def evaluate_individual(individual, models):
    idx = int(individual[0] * (len(models) - 1))  # Scale the float to the index range
    idx = min(max(idx, 0), len(models) - 1)  # Ensure the index is within bounds
    model = models[idx]
    return (model['miss_chance'], model['accuracy'])

def moea(task, models_dir):
    models = predict_ddl_miss_chance(task, models_dir)
    toolbox = base.Toolbox()
    setup_toolbox(models, toolbox)
    population = toolbox.population(n=100)
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, verbose=False)
    best_ind = tools.selBest(population, k=1)[0]
    best_index = int(best_ind[0] * len(models))
    best_index = min(max(best_index, 0), len(models) - 1)  # Ensure the index is within bounds
    best_model = models[best_index]
    task.priority = best_model['miss_chance']
    task.variant = best_model['variant']
    return best_model['variant']

def execute_task(task, models_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_variant = moea(task, models_dir)
    model_path = os.path.join(models_dir, task.model_type, model_variant)
    model = load_model(model_path, device)
    data_loader = load_data_loader(task.dataset, './data', task.batch_size, task.model_type)

    start_time = time.time()
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            _ = model(images)

    elapsed_time = time.time() - start_time
    elapsed_time_ms = elapsed_time * 1000
    task.missed_deadline = elapsed_time_ms > task.deadline

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Task Execution Complete:")
    print(f"  Model Type: {task.model_type}")
    print(f"  Dataset: {task.dataset}")
    print(f"  Deadline: {task.deadline} ms")
    print(f"  Elapsed Time: {elapsed_time_ms:.2f} ms")
    print(f"  Deadline Met: {'No' if task.missed_deadline else 'Yes'}")
    print(f"  Model Variant: {model_variant}\n")

def read_task_definitions(csv_file_path, models_dir):
    tasks = []
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            task = Task(row['model_type'], row['dataset'], row['batch_size'], row['start_time_ms'], row['deadline_ms'])
            heapq.heappush(tasks, task)
    return tasks

def main(csv_file_path, models_dir):
    tasks = read_task_definitions(csv_file_path, models_dir)
    results = []

    while tasks:
        next_task = heapq.heappop(tasks)
        current_time = int(time.time() * 1000)
        start_time = next_task.start_time

        if start_time > current_time:
            wait_time = (start_time - current_time) / 1000
            print(f"Waiting {wait_time:.2f} seconds to start task {next_task.model_type}")
            time.sleep(wait_time)

        execute_task(next_task, models_dir)
        results.append(next_task)

    total_tasks = len(results)
    missed_count = sum(1 for result in results if result.missed_deadline)
    deadline_miss_rate = (missed_count / total_tasks) * 100 if total_tasks > 0 else 0

    print("\nFinal Results:")
    print(f"Total tasks: {total_tasks}")
    print(f"Tasks that met the deadline: {total_tasks - missed_count}")
    print(f"Tasks that missed the deadline: {missed_count}")
    print(f"Deadline Miss Rate: {deadline_miss_rate:.2f}%")

if __name__ == "__main__":
    main('./task_definitions.csv', './models')

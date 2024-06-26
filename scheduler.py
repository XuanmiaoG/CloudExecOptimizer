import csv
import os
import time
import torch
import heapq
import random
import threading
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from deap import base, creator, tools, algorithms

# Define the problem object using DEAP's creator
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # Minimize miss chance, maximize accuracy
creator.create("Individual", list, fitness=creator.FitnessMulti)

class Task:
    def __init__(self, task_id, model_type, dataset, batch_size, start_time, deadline, data_size):
        self.task_id = task_id
        self.model_type = model_type
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.start_time = int(start_time)
        self.deadline = int(deadline)
        self.data_size = int(data_size)
        self.priority = float('inf')
        self.variant = None
        self.missed_deadline = None

    def __lt__(self, other):
        return self.priority < other.priority

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def load_data_loader(data_directory, batch_size, model_type, data_size):
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
    dataset = datasets.CIFAR10(root=data_directory, train=False, download=False, transform=transform)
    indices = list(range(len(dataset)))
    if data_size < len(indices):
        indices = indices[:data_size]
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)

def predict_ddl_miss_chance(task, models_dir):
    models = []
    csv_file_path = os.path.join(models_dir, task.model_type, f"{task.model_type}_inference_results.csv")
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Calculate the inference time for the given data_size
            total_inference_time = float(row['Inference Time (s)'])
            total_data_size = 10000  # Total number of images in the CIFAR-10 test set
            scaled_inference_time = total_inference_time * (task.data_size / total_data_size)
            
            # Scale inference time with batch size
            scaled_inference_time *= (task.batch_size / 100)

            accuracy = float(row['Accuracy (%)'])
            miss_chance = max(0, scaled_inference_time - (task.deadline / 1000))  # Convert deadline to seconds
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

def monitor_scheduler(start_time, task_waitlist, task_scheduler_queue, interval=10):
    while True:
        current_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        while task_waitlist and task_waitlist[0].start_time <= current_time:
            task = heapq.heappop(task_waitlist)
            heapq.heappush(task_scheduler_queue, task)
        num_tasks_remaining = len(task_scheduler_queue)
        print(f"[Scheduler Runtime Info] Current Time: {current_time / 1000:.2f} seconds, Tasks Remaining: {num_tasks_remaining}")
        time.sleep(interval)

def execute_task(task, models_dir, results_file, scheduler_start_time):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_variant = moea(task, models_dir)
    model_path = os.path.join(models_dir, task.model_type, model_variant)
    model = load_model(model_path, device)
    data_loader = load_data_loader('./data', task.batch_size, task.model_type, task.data_size)

    start_time = time.time()
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            _ = model(images)
    elapsed_time = time.time() - (scheduler_start_time + task.start_time/1000)
    elapsed_time_ms = elapsed_time * 1000
    actual_start_time = (start_time - scheduler_start_time) * 1000
    task.missed_deadline = elapsed_time_ms > task.deadline

    results = {
        'task_id': task.task_id,
        'model_type': task.model_type,
        'dataset': task.dataset,
        'batch_size': task.batch_size,
        'start_time': task.start_time,
        'actual_start_time': actual_start_time,
        'deadline': task.deadline,
        'elapsed_time_ms': elapsed_time_ms,
        'missed_deadline': task.missed_deadline,
        'model_variant': model_variant,
        'data_size': task.data_size
    }

    with open(results_file, 'a', newline='') as csvfile:
        fieldnames = ['task_id', 'model_type', 'dataset', 'batch_size', 'start_time', 'actual_start_time', 'deadline', 'elapsed_time_ms', 'missed_deadline', 'model_variant', 'data_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(results)

def read_task_definitions(csv_file_path):
    tasks = []
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            task = Task(row['task_id'], row['model_type'], row['dataset'], row['batch_size'], row['start_time_ms'], row['deadline_ms'], row['data_size'])
            tasks.append(task)
    tasks.sort(key=lambda x: x.start_time)  # Sort tasks by start_time
    return tasks


def main(task_definitions_file, models_dir, results_file):
    task_waitlist = read_task_definitions(task_definitions_file)
    task_scheduler_queue = []
    results = []

    # Initialize results file with headers for the task execution results
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['task_id', 'model_type', 'dataset', 'batch_size', 'start_time', 'actual_start_time', 'deadline', 'elapsed_time_ms', 'missed_deadline', 'model_variant', 'data_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    scheduler_start_time = time.time()

    # Start the monitoring thread
    monitor_thread = threading.Thread(target=monitor_scheduler, args=(scheduler_start_time, task_waitlist, task_scheduler_queue))
    monitor_thread.daemon = True
    monitor_thread.start()

    while task_waitlist or task_scheduler_queue:
        current_time = (time.time() - scheduler_start_time) * 1000  # Convert to milliseconds

        # Move tasks from waitlist to scheduler queue
        while task_waitlist and task_waitlist[0].start_time <= current_time:
            task = heapq.heappop(task_waitlist)
            heapq.heappush(task_scheduler_queue, task)

        # Execute tasks in the scheduler queue
        if task_scheduler_queue:
            next_task = heapq.heappop(task_scheduler_queue)
            execute_task(next_task, models_dir, results_file, scheduler_start_time)
            results.append(next_task)

        # Avoid busy waiting
        time.sleep(0.1)  # Sleep for a short duration

    total_tasks = len(results)
    missed_count = sum(1 for result in results if result.missed_deadline)
    deadline_miss_rate = (missed_count / total_tasks) * 100 if total_tasks > 0 else 0

    summary_results = {
        'total_tasks': total_tasks,
        'tasks_met_deadline': total_tasks - missed_count,
        'tasks_missed_deadline': missed_count,
        'deadline_miss_rate': deadline_miss_rate
    }

    # Append summary statistics to the results file (simulating Sheet 2)
    with open(results_file, 'a', newline='') as csvfile:
        csvfile.write('\n')  # Add an empty line to separate sections
        writer = csv.DictWriter(csvfile, fieldnames=summary_results.keys())
        writer.writeheader()
        writer.writerow(summary_results)

    print("\nFinal Results:")
    print(f"Total tasks: {total_tasks}")
    print(f"Tasks that met the deadline: {total_tasks - missed_count}")
    print(f"Tasks that missed the deadline: {missed_count}")
    print(f"Deadline Miss Rate: {deadline_miss_rate:.2f}%")

if __name__ == "__main__":
    main('./task_definitions1.csv', './models', './results.csv')


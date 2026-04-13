# modified from https://github.com/saprmarks/geometry-of-truth
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

from utils import DataManager
from probes import MMProbe

# ##############################################################################
# ## CONFIGURATION
# ##############################################################################
# Group all hyperparameters and settings here for easy modification.

CONFIG = {
    'model': 'llama-3.1-8b',
    'layers_to_test': [12],
    'seeds_to_test': range(1), # Run experiment across multiple seeds
    'ProbeClass': MMProbe,
    'act_types': ['latents', 'centroids'],
    'data_split': 0.8,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'noperiod': False,
    'center_data': True,
    
    # Paths for saving results and plots
    'output_dir': Path('outputs/probes'),
    'results_file': Path('outputs/probes/raw_probe_results.json'),

    'train_medleys': [
        ['likely']
    ],

    'val_datasets': [
        'cities', 'neg_cities', 'larger_than', 'smaller_than', 'sp_en_trans', 
        'neg_sp_en_trans', 'cities_cities_conj', 'cities_cities_disj', 
        'companies_true_false', 'common_claim_true_false', 'counterfact_true_false', 
    ]
}

# ##############################################################################
# ## HELPER FUNCTIONS for RESULTS IO
# ##############################################################################

def load_results(filepath):
    """Loads results from a JSON file if it exists."""
    if filepath.exists():
        with open(filepath, 'r') as f:
            print(f"Loading existing results from {filepath}")
            return json.load(f, object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()})
    return {}

def save_results(data, filepath):
    """Saves results to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    # print(f"Results saved to {filepath}")

# ##############################################################################
# ## CORE LOGIC
# ##############################################################################

def train_and_evaluate(layer, act_type, train_medley, val_datasets, seed, config):
    """
    Trains a single probe for a given seed and evaluates it.
    """
    # Set seed for this specific run for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    dm = DataManager(act_type=act_type)
    
    # Load training & validation data
    for dataset in train_medley:
        dm.add_dataset(dataset, config['model'], layer, split=config['data_split'], 
                       seed=seed, noperiod=config['noperiod'], 
                       center=config['center_data'], device=config['device'])
    for dataset in val_datasets:
        if dataset not in train_medley:
            dm.add_dataset(dataset, config['model'], layer, split=None, 
                           noperiod=config['noperiod'], center=config['center_data'], 
                           device=config['device'])

    # Train probe
    train_acts, train_labels = dm.get('train')
    probe = config['ProbeClass'].from_data(train_acts, train_labels, device=config['device'])

    # Evaluate probe
    accuracies = {}
    for val_dataset in val_datasets:
        is_iid = val_dataset in train_medley
        acts, labels = dm.data['val'][val_dataset] if is_iid else dm.data[val_dataset]
        acc = (probe.pred(acts, iid=is_iid) == labels).float().mean().item()
        accuracies[val_dataset] = acc
        
    return accuracies

def aggregate_results(all_results):
    """
    Calculates the mean and standard deviation of accuracies across seeds.
    """
    if not all_results: return {}
    
    # Use the structure of the first seed's data as a template
    seeds = list(all_results.keys())
    template = all_results[seeds[0]]
    aggregated = {layer: {act: {} for act in data.keys()} for layer, data in template.items()}

    for layer in template.keys():
        for act_type in template[layer].keys():
            for medley_name in template[layer][act_type].keys():
                val_datasets = template[layer][act_type][medley_name].keys()
                
                means = {}
                stds = {}
                
                for val_ds in val_datasets:
                    # Collect scores for this specific validation set from all seeds
                    scores = [all_results[seed][layer][act_type][medley_name][val_ds]
                              for seed in seeds if val_ds in all_results[seed].get(layer, {}).get(act_type, {}).get(medley_name, {})]
                    
                    means[val_ds] = np.mean(scores) if scores else 0
                    stds[val_ds] = np.std(scores) if scores else 0
                
                aggregated[layer][act_type][medley_name] = {'means': means, 'stds': stds}
                
    return aggregated

def plot_with_error_bars(aggregated_data, layer, train_medley_name, config):
    """
    Generates and saves a bar chart with mean accuracies and error bars.
    """
    try:
        acts_data = aggregated_data['latents'][train_medley_name]
        cents_data = aggregated_data['centroids'][train_medley_name]
    except KeyError:
        print(f"Warning: Missing aggregated data for '{train_medley_name}' at layer {layer}. Skipping plot.")
        return

    val_datasets = list(acts_data['means'].keys())
    x_labels = [name.replace('_', ' ').title() for name in val_datasets]
    x = np.arange(len(val_datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained',dpi=200)
    
    # Plot bars with error bars (yerr)
    ax.bar(x - width/2, acts_data['means'].values(), width, label='Latents', capsize=5, color=plt.cm.autumn(0.25))
    ax.bar(x + width/2, cents_data['means'].values(), width, label='Centroids', capsize=5, color=plt.cm.winter(0.25))

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xticks(x, x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    output_path = config['output_dir'] / f"L{layer}_{train_medley_name}_avg.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved aggregated plot to {output_path}")
    plt.close(fig)

# ##############################################################################
# ## MAIN EXECUTION
# ##############################################################################

def main():
    """
    Main function to run the entire suite of experiments and generate plots.
    """
    all_results = load_results(CONFIG['results_file'])

    # --- Data Collection Phase ---
    for seed in CONFIG['seeds_to_test']:
        if seed not in all_results:
            all_results[seed] = {}
        
        for layer in CONFIG['layers_to_test']:
            if layer not in all_results[seed]:
                all_results[seed][layer] = {}

            for act_type in CONFIG['act_types']:
                if act_type not in all_results[seed][layer]:
                    all_results[seed][layer][act_type] = {}

                for medley in CONFIG['train_medleys']:
                    medley_name = '+'.join(medley)
                    
                    # Check if this run already exists
                    if medley_name in all_results[seed][layer][act_type]:
                        print(f"Skipping existing run: Seed {seed}, Layer {layer}, Type '{act_type}', Train on '{medley_name}'")
                        continue

                    print(f"--- Running: Seed {seed}, Layer {layer}, Type '{act_type}', Train on '{medley_name}' ---")
                    accuracies = train_and_evaluate(
                        layer, act_type, medley, CONFIG['val_datasets'], seed, CONFIG
                    )
                    all_results[seed][layer][act_type][medley_name] = accuracies
        
        # Save after each seed is fully processed
        save_results(all_results, CONFIG['results_file'])
        print(f"Completed and saved results for seed {seed}.")

    # --- Aggregation and Plotting Phase ---
    print("\n--- Aggregating results and generating plots ---")
    aggregated_results = aggregate_results(all_results)
    
    if not aggregated_results:
        print("No results to plot. Exiting.")
        return

    for layer in aggregated_results.keys():
        for medley in CONFIG['train_medleys']:
            medley_name = '+'.join(medley)
            plot_with_error_bars(aggregated_results[layer], layer, medley_name, CONFIG)

if __name__ == "__main__":
    main()
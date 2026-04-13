# modified from https://github.com/saprmarks/geometry-of-truth
import torch as t
import argparse
import pandas as pd
from tqdm import tqdm
import os
import configparser
from nnsight import LanguageModel
import torch
import time
import pickle

DEBUG = False
if DEBUG:
    tracer_kwargs = {'scan': True, 'validate': True}
else:
    tracer_kwargs = {'scan': False, 'validate': False}

config = configparser.ConfigParser()
config.read('probes/config.ini')

def load_model(model_name, device='remote'):
    print(f"Loading model {model_name}...")
    weights_directory = config[model_name]['weights_directory']
    if device == 'remote':
        model = LanguageModel(weights_directory)
    else:
        model = LanguageModel(weights_directory, dtype=t.bfloat16, device_map="auto")
    return model

def load_statements(dataset_name):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(f"probes/datasets/{dataset_name}.csv")
    statements = dataset['statement'].tolist()
    return statements

def get_cents(statements, model, layer, remote=True):
    """
    Get given layer activations for the statements. 
    Return dictionary of stacked activations.
    """
    start_time=time.time()
    inps = {}
    with model.trace(statements, remote=remote, **tracer_kwargs):
        inps[layer] = model.model.layers[layer].mlp.input[:,-1,:].save().cpu()

    acts={}
    t.set_grad_enabled(True)
    for layer, inp in inps.items():
        inp.requires_grad_().to(model.device)
        out=model.model.layers[layer].mlp(inp)
        cent=torch.autograd.grad(out,inp,torch.ones_like(out))[0]
        acts[layer] = cent
    t.set_grad_enabled(False)
    end_time=time.time()
    return acts, end_time-start_time

def get_acts(statements, model, layer, remote=True):
    """
    Get given layer activations for the statements. 
    Return dictionary of stacked activations.
    """
    start_time=time.time()
    acts = {}
    with model.trace(statements, remote=remote, **tracer_kwargs):
        acts[layer] = model.model.layers[layer].output[:,-1,:].save().cpu()

    for layer, act in acts.items():
        acts[layer] = act
    end_time=time.time()
    return acts, end_time-start_time

if __name__ == "__main__":
    """
    read statements from dataset, record activations in given layers, and save to specified files
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama-3.1-8b",)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--datasets", nargs='+')
    parser.add_argument("--output_dir", default="outputs/probes")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    t.set_grad_enabled(False)
    model = load_model(args.model, args.device)
    dataset_times={}
    for dataset in args.datasets:
        statements = load_statements(dataset)
        layers = args.layers
        if layers == [-1]:
            layers = list(range(len(model.model.layers)))
        centroid_save_dir = os.path.join(f"{args.output_dir}", "centroids", dataset)
        latent_save_dir = os.path.join(f"{args.output_dir}", "latents", dataset)
        if not os.path.exists(centroid_save_dir):
            os.makedirs(centroid_save_dir)
        if not os.path.exists(latent_save_dir):
            os.makedirs(latent_save_dir)

        centroid_time=0
        latent_time=0
        for idx in tqdm(range(0, len(statements), 25)):
            acts, lt = get_acts(statements[idx:idx + 25], model, layers, args.device == 'remote')
            cents, ct = get_cents(statements[idx:idx + 25], model, layers, args.device == 'remote')
            centroid_time+=ct
            latent_time+=lt
            for layer, cent in cents.items():
                    t.save(cent, f"{centroid_save_dir}/layer_{layer}_{idx}.pt")
            for layer, act in acts.items():
                    t.save(act, f"{latent_save_dir}/layer_{layer}_{idx}.pt")
        end_time=time.time()
        dataset_times[dataset]={'Centroids':centroid_time,'Latents':latent_time}
    with open(f'{args.output_dir}/times.pkl','wb') as file:
        pickle.dump(dataset_times,file)
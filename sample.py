'''
Test trained GenViT baseline for VF prediction.
'''

import argparse
from diffusion import GaussianDiffusion
import torch
import json
import numpy as np
import statistics
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pickle
from PIL import Image

from VL_utils import VF_Dataset, GENVIT, scale_data, masked_loss, clean_vf, remove_subplot_lines
from models.DifViT import ViT as DifViT

def sample(model, input, gpu, normalize, num_samples):
    # gpu
    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # sampling
    clamp = True
    samples = []

    with torch.no_grad():
        for _ in tqdm(range(num_samples)):
            img, age, horizon, gender, side = input
            img = img.to(device).float()
            age, horizon, gender, side = age.to(device).float(), horizon.to(device).float(), gender.to(device), side.to(device)

            # outputs = model(img, age, horizon, gender, side)
            outputs = model.sample(img, age, horizon, gender, side, clamp=clamp)

            if normalize:
                outputs = scale_data(outputs,0,1,dataset.data_range[0],dataset.data_range[1])

            samples_array = outputs.detach().cpu().numpy()
            samples.extend(samples_array)

    return np.array(samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tr_dataset', type=str, choices=['uwhvf','uwhvf_extended','uwhvf_random', 'uwhvf_int_random', 'scheie', 'scheie_random'], default='uwhvf_extended', help='Dataset to train on.')
    parser.add_argument('--dataset', type=str, choices=['uwhvf','uwhvf_extended','uwhvf_random', 'uwhvf_int_random', 'scheie', 'scheie_random', 'uwhvf_int'], default='uwhvf_extended', help='Dataset to train on.')
    parser.add_argument('--representation', type=str, choices=['hvf','td'], default='td', help='Representation of data.')
    parser.add_argument('--model_pth', type=str, help='Path of trained model.')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Directory to save training results in.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU.')
    parser.add_argument('--plot_all_batches', action='store_true', help='Use arg to plot all batches.')
    parser.add_argument('--cached', type=str, default=None, help='Supply path of cached results.')

    # details
    parser.add_argument('--num_samples', type=int, default=200, help='Number of samples for each image.')
    parser.add_argument('--num_vf', type=int, default=1500, help='Number of visual fields to sample for.')
    parser.add_argument('--target', type=str, choices=['x0', 'noise'], default='x0', help='Which target to use when training.')
    parser.add_argument('--normalize', action='store_true', help='Use arg to normalize data.')
    parser.add_argument('--cond_age', action='store_true', help='Use arg to condition on age.')
    parser.add_argument('--cond_horizon', action='store_true', help='Use arg to condition on prediction horizon.')
    parser.add_argument('--cond_gender', action='store_true', help='Use arg to condition on gender.')
    parser.add_argument('--cond_side', action='store_true', help='Use arg to condition on side of eye.')
    
    args = parser.parse_args()
    print(args)
    
    args.save_dir = f'{args.save_dir}'
    if args.target == 'noise':
        args.normalize = True
    
    # seeds
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)  

    # load data
    print('Loading data...')
    if args.dataset == 'uwhvf_int':
        data_dir = f'../samples/GP-Detection-VF-Prediction/Diffusion_ViT_uwhvf/datasets/uwhvf_int_processed'
    else:
        data_dir = f'datasets/{args.dataset}_processed'
    with open(os.path.join(data_dir, 'test.json')) as f:
        test_data = json.loads(f.read())
    if args.representation == 'hvf':
        data_range = (0,50)
    elif (args.representation == 'td') and ('uwhvf' in args.dataset):
        data_range = (-37.69, 50.00)
    elif (args.representation == 'td') and ('scheie' in args.dataset):
        data_range = (-38.00, 40.00)
    else:
        raise NotImplementedError
    test_dataset = VF_Dataset(test_data, args.representation, args.normalize, data_range)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.num_vf, shuffle=False)

    input, labels, _ = next(iter(test_dataloader))
    
    valid = []
    for i in range(args.num_vf):
        md = torch.mean(input[0][i], dtype=torch.float64).item()
        if md < -6:
            valid.append(i)
            
    input = [input[i][valid] for i in range(len(input))]
    labels = labels[valid]
    
    args.num_vf = len(input[0])
    print(args.num_vf)
    
    print('Loading model...')
    model = GENVIT(conditional={
                'age':args.cond_age,
                'horizon':args.cond_horizon,
                'gender':args.cond_gender,
                'side':args.cond_side
            }, target=args.target,
            data_range=data_range)
    model.load_state_dict(torch.load(args.model_pth))
    model.eval()

    # test loop
    print('Sampling...')
    samples = sample(model, input, args.gpu, args.normalize, args.num_samples)

    num_mild = 0
    num_mod = 0
    num_severe = 0

    for i in range(args.num_vf):
        md = torch.mean(input[0][i], dtype=torch.float64).item()
        if md > -6:
            subgroup = 'mild'
            index = num_mild
            num_mild += 1
        elif md > -12:
            subgroup = 'moderate'
            index = num_mod
            num_mod += 1
        else:
            subgroup = 'severe'
            index = num_severe
            num_severe += 1
        img_path = f'{args.save_dir}/{subgroup}/ground_truths'
        os.makedirs(img_path, exist_ok=True)
        img = Image.fromarray(clean_vf(np.array(labels[i]), test_data['padding_mask'], data_range, 1)
                              .astype(np.uint8))
        img.save(os.path.join(img_path, f'{index}.png'))

    num_mild = 0
    num_mod = 0
    num_severe = 0

    for i in range(args.num_samples):
        for j in range(args.num_vf):
            md = torch.mean(input[0][j], dtype=torch.float64).item()
            if md > -6:
                subgroup = 'mild'
                index = num_mild
                num_mild += 1
            elif md > -12:
                subgroup = 'moderate'
                index = num_mod
                num_mod += 1
            else:
                subgroup = 'severe'
                index = num_severe
                num_severe += 1
            img_path = f'{args.save_dir}/{subgroup}/samples/{index}'
            os.makedirs(img_path, exist_ok=True)
            img = Image.fromarray(clean_vf(np.array(samples[i * args.num_vf + j]), test_data['padding_mask'], data_range, 1)
                                .astype(np.uint8))
            img.save(os.path.join(img_path, f'{i}.png'))
            
    print(f'Output: {num_mild} mild, {num_mod} moderate, {num_severe} severe')
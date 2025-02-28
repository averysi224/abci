import os
import logging
import argparse
import random
import pdb

import torch
from torchvision.transforms import transforms as T
import matplotlib.pyplot as plt
import pandas as pd

from core import DAPUQUncertaintyRegion
from puq.arch_sample import hvf_json
from data.data import DiffusionSamplesDataset, GroundTruthsDataset, DiffusionSamplesDataLoader, GroundTruthsDataLoader
from plotting.visual import plot_archetype_matrices
from utils import misc

def get_arguments():
    parser = argparse.ArgumentParser(description='Official implementation of "Principal Uncertainty Quantification with Spatial Correlation for Image Restoration Problems" paper. link: https://arxiv.org/abs/2305.10124')
    parser.add_argument('--method', type=str, default='da_puq', help='Method to use: e_puq, da_puq or rda_puq.')
    parser.add_argument('--data', type=str, default=f'data/UW_subgroups/moderate', help='Data folder path.')

    parser.add_argument('--test-ratio', type=float, default=0.2, help='Test instances ratio out of the data folder.')
    parser.add_argument('--seed', type=int, default=42, help='Seed.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use, None for CPU.')
    parser.add_argument('--batch', type=int, default=4, help='Batch size to work on.')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for dataloaders.')
    parser.add_argument('--no-cache', action='store_true', default=False, help='')

    parser.add_argument('--alpha', type=float, default=0.25, help='Coverage guarantee parameter.')
    parser.add_argument('--beta', type=float, default=0.14, help='Reconstruction guarantee parameter.')
    parser.add_argument('--q', type=float, default=0.9, help='Pixels ratio parameter for reconstruction guarantee.')
    parser.add_argument('--delta', type=float, default=0.1, help='Error level of guarantees.')

    # Technical parameters for calibration procedure
    parser.add_argument('--num-reconstruction-lambdas', type=int, default=17, help='Number of fine-grained lambdas parameters to be checked for reconstruction guarantee.')        # lambda1s
    parser.add_argument('--num-coverage-lambdas', type=int, default=100, help='Number of fine-grained lambdas parameters to be checked for coverage guarantee.')                    # lambda2s
    parser.add_argument('--num-pcs-lambdas', type=int, default=20, help='Number of fine-grained lambdas parameters to be checked for reducting the number of PCs at inference.')    # lambda3s
    parser.add_argument('--max-coverage-lambda', type=float, default=800.0, help='Maximal coverage lambda. (20.0 should be enougth for various tasks).')

    # Archetypes
    parser.add_argument('--archetypes', action='store_true', default=False, help='Archetypes')

    args = parser.parse_args()
    return args

def plot_vf(sample):
    '''Convert alg outputs to plotting format.'''
    coordinates = [(entry["x"], entry["y"]) for entry in hvf_json]
    vf = [1.0 for _ in range(81)]
    vf[34] = 0.0
    vf[43] = 0.0
    for j in range(len(coordinates)):
        vf[coordinates[j][0] * 9 + coordinates[j][1]] = sample[coordinates[j][0] * 9 + coordinates[j][1]].item()
    return torch.Tensor(vf)

def main(args):

    misc.setup_logging(os.path.join(os.getcwd(), 'log.txt'))
    logging.info(args)
    torch.manual_seed(args.seed)

    if not os.path.exists("results"):
        os.makedirs("results")

    puq = DAPUQUncertaintyRegion(args)
    
    # CALIBRATION
    # load pre-generated visual field predictions
    cal_samples_dataset = DiffusionSamplesDataset(
        opt=args,
        calibration=True,
        transform=T.Compose([T.Grayscale(num_output_channels=1), T.ToTensor()])
    )
    # load pre-generated ground truth
    cal_ground_truths_dataset = GroundTruthsDataset(
        opt=args,
        samples_dataset=cal_samples_dataset,
        transform=T.Compose([T.Grayscale(num_output_channels=1), T.ToTensor()])
    )

    puq.calibration(cal_samples_dataset, cal_ground_truths_dataset)
    
    test_samples_dataset = DiffusionSamplesDataset(
        opt=args,
        calibration=False,
        transform=T.Compose([T.Grayscale(num_output_channels=1), T.ToTensor()])
    )

    test_ground_truths_dataset = GroundTruthsDataset(
        opt=args,
        samples_dataset=test_samples_dataset,
        transform=T.Compose([T.Grayscale(num_output_channels=1), T.ToTensor()])
    )
    # get quantitative results
    results = puq.eval(test_samples_dataset, test_ground_truths_dataset)

    # load test dataset
    dl = DiffusionSamplesDataLoader(
        test_samples_dataset,
        batch_size=args.batch,
        num_workers=args.num_workers
    )

    gl = GroundTruthsDataLoader(
        test_ground_truths_dataset,
        batch_size=args.batch,
        num_workers=args.num_workers
    )

    dl = iter(dl)
    gl = iter(gl)

    all_diff = 0
    cnt = 0

    # Load HVF archetypes for plotting
    archetypes = pd.read_csv("./plotting/at17_matrix.csv").values  

    for i_batch in range(len(test_ground_truths_dataset) // 4):
        image_shape = [1, 9, 9]
        batch_samples = next(dl).flatten(2)
        batch_gt = next(gl).flatten(2).squeeze()
        batch_gt = [plot_vf(i.cpu()) for i in batch_gt]

        # get the average prediction, high-contributing archetypes (ordered), bounds and selected archetypes indices for test set
        batch_mu, batch_pcs, batch_svs, batch_lower, batch_upper, batch_indices = puq.inference(batch_samples)
        batch_mu = [plot_vf(i.cpu()) for i in batch_mu]
        batch_pcs = [i.cpu() for i in batch_pcs]
        batch_svs = [i.cpu() for i in batch_svs]
        batch_lower = [i.cpu() for i in batch_lower]
        batch_upper = [i.cpu() for i in batch_upper]
        batch_indices = [i.cpu() for i in batch_indices]

        fig, axs = plt.subplots(args.batch, 8, figsize=(20, 10))

        for i in range(args.batch):
            axs[i, 0].imshow(batch_mu[i].view(image_shape).transpose(0,1).transpose(1,2), cmap='gray')
            axs[i, 1].axis('off')

            lower_image = (batch_mu[i] + batch_pcs[i] @ batch_lower[i]).clamp_(0, 1)
            axs[i, 1].imshow(lower_image.view(image_shape).transpose(0,1).transpose(1,2), cmap='gray')
            axs[i, 1].axis('off')

            upper_image = (batch_mu[i] + batch_pcs[i] @ batch_upper[i]).clamp_(0, 1)
            axs[i, 2].imshow(upper_image.view(image_shape).transpose(0,1).transpose(1,2), cmap='gray')
            axs[i, 2].axis('off')

            axs[i, 3].imshow(batch_gt[i].view(image_shape).transpose(0,1).transpose(1,2), cmap='gray')
            axs[i, 3].axis('off')

            diff = torch.abs(upper_image - lower_image).sum() / 52.
            all_diff += diff
            cnt += 1

            # archetype 0 is not considered as visual loss
            selected_pcs = batch_indices[i]
            selected_pcs = selected_pcs[selected_pcs != 0]

            # plot the top-4 high-contributiong archetypes
            for axis_i in range(4):
                if batch_pcs[i].shape[1] > axis_i:
                    plot_archetype_matrices(axs[i, axis_i+4], selected_pcs[axis_i], archetypes)
                axs[i, axis_i+4].axis('off')

        cols = ['average\nprediction', 'lower\nbound', 'upper\nbound', 'ground\ntruth', '', '', 'main uncertainty components\n', '']
        for ax, col in zip(axs[0], cols):
            ax.set_title(col, fontsize=20, pad=20)

        dis = [0.93, 0.8, 0.6, 0.4, 0.2]
        # Add 4x1 text outside the grid
        texts = ["patient", "   1   ", "   2   ", "   3   ", "   4   "]
        for row_idx, text in enumerate(texts):
            fig.text(
                x=0.05,  # Position horizontally
                y=dis[row_idx],  # Position vertically, normalized to the row
                s=text,
                va="center",  # Vertically align the text within the space
                ha="left",  # Align text to the left horizontally
                fontsize=20
            )

        stage = args.data.split('/')[-1]
        fig.suptitle('', fontsize=20, y=0.95)
        plt.savefig(f'results/{stage}_{args.alpha}_bounds_{i_batch}.png')
        plt.close()

    print("interval size: ", all_diff.item()/cnt)
        

if __name__ == "__main__":
    args = get_arguments()
    main(args)

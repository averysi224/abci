import logging
from tqdm import tqdm
from pathlib import Path

import torch
import numpy as np
import pdb


import puq.metrics as metrics
from puq.data.data import DiffusionSamplesDataset, GroundTruthsDataset, DiffusionSamplesDataLoader, GroundTruthsDataLoader
from puq.utils import statistics, misc

import archetypes as arch
from arch_sample import load_archetype_matrix, normalize_data, decompose_hvf_data


class PUQUncertaintyRegion:
    '''
    Base class for PUQ procedures.
    '''

    def __init__(self, opt):
        self.opt = opt
        self.calibrated = False
        self.max_pcs = None

        if self.opt.archetypes:
            archetype_matrix = load_archetype_matrix('data/archetypes.csv')
            norms = np.linalg.norm(archetype_matrix, axis=1)[...,None]  # Shape: (17, 1)

            # # Normalize each vector
            self.archetype_matrix = archetype_matrix / norms            

            self.aa = arch.AA(n_archetypes = 17)
            self.aa.archetypes_ = self.archetype_matrix
            
        self.device = torch.device(
            f'cuda:{self.opt.gpu}' if self.opt.gpu is not None else 'cpu')

    def _init_lambdas(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_batch_losses(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_p_values(self, *args, **kwargs):
        raise NotImplementedError

    def _calibration_failed_message(self, *args, **kwargs):
        raise NotImplementedError

    def _assign_lambdas(self, *args, **kwargs):
        raise NotImplementedError

    def _apply_lambdas(self, *args, **kwargs):
        raise NotImplementedError

    def _eval_metrics(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_intervals(self, samples, eval=False):
        if self.max_pcs is not None and samples.shape[1] > self.max_pcs:
            rand_indices = torch.randperm(samples.shape[1])[:self.max_pcs]
            samples = samples[:, rand_indices]

        # Conditional mean
        mu = samples.mean(dim=1)

        # Principal componenets and singular values (importance weights)
        samples_minus_mean = samples - mu.unsqueeze(1)
        archetype_basis = torch.from_numpy(np.repeat(self.archetype_matrix[np.newaxis, :, :],
                                               samples.shape[0], axis=0)).float().to(self.device)

        projections = torch.bmm(samples_minus_mean.to(self.device), archetype_basis.mT)
        coefficients = torch.norm(projections, dim=1).to(self.device)

        lower = torch.quantile(projections, self.opt.alpha/2, dim=1)
        upper = torch.quantile(projections, 1-(self.opt.alpha/2), dim=1)
        # reorder from large to small
        indices = coefficients.argsort(dim=1, descending=True)
        
        coefficients = torch.gather(coefficients, dim=1, index=indices)
        archetype_basis = torch.gather(archetype_basis, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, 81))
        lower = torch.gather(lower, dim=1, index=indices)
        upper = torch.gather(upper, dim=1, index=indices)

        if eval:
            # indices are the number of the archetypes
            return mu, archetype_basis.mT, coefficients, lower, upper, indices
        else:
            return mu, archetype_basis.mT, coefficients, lower, upper

    def approximation(self, dataset: DiffusionSamplesDataset, verbose=True):
        if verbose:
            logging.info('Applying approximation phase...')

        K = dataset.num_samples_per_image

        dataloader = DiffusionSamplesDataLoader(
            dataset,
            batch_size=self.opt.batch,
            num_workers=self.opt.num_workers
        )

        all_mu = []
        all_pcs = []
        all_svs = []
        all_lower = []
        all_upper = []

        dataloader = tqdm(dataloader) if verbose else dataloader
        for samples in dataloader:
            samples = samples.flatten(2).to(self.device)
            mu, pcs, svs, lower, upper = self._compute_intervals(samples)
            
            all_mu.append(mu.cpu())
            all_pcs.append(pcs.cpu())
            all_svs.append(svs.cpu())
            all_lower.append(lower.cpu())
            all_upper.append(upper.cpu())

        all_mu = torch.cat(all_mu, dim=0)
        all_pcs = torch.cat(all_pcs, dim=0)
        all_svs = torch.cat(all_svs, dim=0)
        all_lower = torch.cat(all_lower, dim=0)
        all_upper = torch.cat(all_upper, dim=0)
        
        return all_mu, all_pcs, all_svs, all_lower, all_upper

    def _compute_losses(self, samples_dataset: DiffusionSamplesDataset, ground_truths_dataset: GroundTruthsDataset, verbose=True):
        all_mu, all_pcs, all_svs, all_lower, all_upper = self.approximation(
            samples_dataset, verbose=verbose)

        dataloader = GroundTruthsDataLoader(
            ground_truths_dataset,
            batch_size=self.opt.batch,
            num_workers=self.opt.num_workers
        )
        dataloader = iter(dataloader)

        num_images = all_mu.shape[0]
        step = self.opt.batch

        all_losses = []
        for i in range(0, num_images, step):
            mu = all_mu[i:i+step].to(self.device)
            pcs = all_pcs[i:i+step].to(self.device)
            svs = all_svs[i:i+step].to(self.device)
            lower = all_lower[i:i+step].to(self.device)
            upper = all_upper[i:i+step].to(self.device)

            # Project ground truths
            ground_truths = next(dataloader).flatten(1).to(self.device)

            ground_truths_minus_mean = ground_truths - mu
            projected_ground_truths_minus_mean = torch.bmm(ground_truths_minus_mean.unsqueeze(1), pcs)[:, 0]

            # Compute losses
            losses = self._compute_batch_losses(
                pcs, svs, lower, upper, ground_truths_minus_mean, projected_ground_truths_minus_mean)
            
            all_losses.append(losses.cpu())
        all_losses = torch.cat(all_losses, dim=1)

        return all_losses

    def calibration(self, samples_dataset: DiffusionSamplesDataset, ground_truths_dataset: GroundTruthsDataset, verbose=True):
        cache_dir_path = f'cache/{self.opt.method}_{self.opt.data.replace("/","_")}_seed{self.opt.seed}_test{self.opt.test_ratio}_alpha{self.opt.alpha}_beta{self.opt.beta}_q{self.opt.q}'
        cache_dir_path = cache_dir_path + cache_dir_path

        loss_table = self._compute_losses(
            samples_dataset, ground_truths_dataset, verbose=verbose)
        if not self.opt.no_cache:
            Path.mkdir(Path(cache_dir_path), exist_ok=True, parents=True)
            torch.save(loss_table, f'{cache_dir_path}/loss_table.pt')

        if verbose:
            logging.info('Applying calibration phase...')

        # Compute risks - reconstruction loss, coverage loss
        risks = loss_table.mean(dim=1)
        # Compute p-values
        pvals = self._compute_p_values(risks, samples_dataset.num_images)

        # Find valid lambdas using bonferroni correction
        valid_indices = statistics.bonferroni_search(pvals, self.opt.delta, downsample_factor=20)
        

        # Assign lambdas that minimize the uncertainty volume
        if valid_indices.shape[0] == 0:
            misc.handle_error(self._calibration_failed_message(
                samples_dataset.num_images))

        self._assign_lambdas(valid_indices, verbose=verbose)
        self.max_pcs = samples_dataset.num_samples_per_image
        self.calibrated = True

    def inference(self, samples):
        assert self.calibrated

        if samples.shape[1] != self.max_pcs:
            misc.handle_error(
                f'PUQ was calibrated for K={self.max_pcs} samples per image, but {self.max_pcs} samples were given.')
        
        mu, pcs, svs, lower, upper, indices = self._compute_intervals(samples, eval=True)
        mu, pcs, svs, lower, upper = self._apply_lambdas(mu, pcs, svs, lower, upper)
        return mu, pcs, svs, lower, upper, indices

    def eval(self, samples_dataset: DiffusionSamplesDataset, ground_truths_dataset: GroundTruthsDataset, verbose=True):
        assert self.calibrated
        
        if verbose:
            logging.info('Applying evaluation...')

        K = samples_dataset.num_samples_per_image

        samples_dataloader = DiffusionSamplesDataLoader(
            samples_dataset,
            batch_size=self.opt.batch,
            num_workers=self.opt.num_workers
        )

        ground_truths_dataloader = GroundTruthsDataLoader(
            ground_truths_dataset,
            batch_size=self.opt.batch,
            num_workers=self.opt.num_workers
        )

        results_list = []

        dataloader = zip(samples_dataloader, ground_truths_dataloader)
        dataloader = tqdm(dataloader, total=len(
            ground_truths_dataloader)) if verbose else dataloader
        for samples, ground_truths in dataloader:

            samples = samples.flatten(2).to(self.device)
            ground_truths = ground_truths.flatten(1).to(self.device)

            mu, pcs, svs, lower, upper = self._compute_intervals(samples)

            ground_truths_minus_mean = ground_truths - mu
            projected_ground_truths_minus_mean = torch.bmm(
                ground_truths_minus_mean.unsqueeze(1), pcs)[:, 0]

            results_list.append(
                self._eval_metrics(mu, pcs, svs, lower, upper, ground_truths_minus_mean, projected_ground_truths_minus_mean))

        results = {}
        for c in results_list[0]:
            metric = 0
            for d in results_list:
                metric += d[c]
            metric /= len(results_list)
            results[c] = metric
        
        if verbose:
            logging.info(results)
        
        return results

class DAPUQUncertaintyRegion(PUQUncertaintyRegion):
    '''
    Implementation for Dimension-Adaptive-PUQ (DA-PUQ) procedure.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconstruction_lambda = None
        self.coverage_lambda = None

    def _init_lambdas(self):
        lambda1s = torch.linspace(
            0, 1, self.opt.num_reconstruction_lambdas).to(self.device)
        lambda2s = torch.linspace(
            1, self.opt.max_coverage_lambda, self.opt.num_coverage_lambdas).to(self.device)
        return lambda1s, lambda2s

    def _compute_batch_losses(self, pcs, svs, lower, upper, ground_truths_minus_mean, projected_ground_truths_minus_mean):
        lambdas = self._init_lambdas()
        lambda1s, lambda2s = lambdas[:2]
        reconstruction_losses = metrics.compute_reconstruction_loss(
            pcs, svs, ground_truths_minus_mean, projected_ground_truths_minus_mean, lambda1s, pixel_ratio=self.opt.q)
        reconstruction_losses = reconstruction_losses.unsqueeze(
            2).repeat(1, 1, lambda2s.shape[0])
        coverage_losses = metrics.compute_coverage_loss(
            svs, lower, upper, projected_ground_truths_minus_mean, lambda1s, lambda2s)
        
        return torch.stack([reconstruction_losses, coverage_losses], dim=0)

    def _compute_p_values(self, risks, num_images):
        pval1s = np.array([statistics.hb_p_value(r_hat, num_images, self.opt.beta)
                           for r_hat in risks[0].flatten().cpu()])
        pval2s = np.array([statistics.hb_p_value(r_hat, num_images, self.opt.alpha)
                           for r_hat in risks[1].flatten().cpu()])
        
        pvals = np.maximum(pval1s, pval2s)
        pvals = np.flip(np.flip(pvals.reshape(
            risks.shape[1], risks.shape[2]), axis=0), axis=1).reshape(-1)
        return pvals

    def _calibration_failed_message(self, num_images):
        return f'Calibration failed: cannot guarantee alpha={self.opt.alpha}, beta={self.opt.beta}, q={self.opt.q}, delta={self.opt.delta} using n={num_images} calibration instances.'

    def _assign_lambdas(self, valid_indices, verbose=True):
        lambda1s, lambda2s = self._init_lambdas()
        
        valid_lambdas = torch.cartesian_prod(lambda1s.flip(0), lambda2s.flip(0))[valid_indices]
        # pdb.set_trace()
        lambda1, _ = valid_lambdas[-1]
        lambda2 = valid_lambdas[valid_lambdas[:, 0] == lambda1, 1].min()

        if verbose:
            logging.info(
                f'Successfully calibrated: lambda1={lambda1}, lambda2={lambda2}')

        self.reconstruction_lambda = lambda1.item()
        self.coverage_lambda = lambda2.item()

    def _apply_lambdas(self, mu, pcs, svs, lower, upper, reconstruction_lambda=None, coverage_lambda=None):
        reconstruction_lambda = reconstruction_lambda if reconstruction_lambda is not None else self.reconstruction_lambda
        coverage_lambda = coverage_lambda if coverage_lambda is not None else self.coverage_lambda
        pcs_masks = metrics.compute_pcs_masks(
            svs, torch.tensor([reconstruction_lambda], device=svs.device))
        pcs_masks = pcs_masks[:, :, 0]
        pcs = [pcs[i, :, :mask.sum()] for i, mask in enumerate(pcs_masks)]
        svs = [svs[i, :mask.sum()] for i, mask in enumerate(pcs_masks)]
        lower = [coverage_lambda * lower[i, :mask.sum()]
                 for i, mask in enumerate(pcs_masks)]
        upper = [coverage_lambda * upper[i, :mask.sum()]
                 for i, mask in enumerate(pcs_masks)]

        return mu, pcs, svs, lower, upper

    def _eval_metrics(self, mu, pcs, svs, lower, upper, ground_truths_minus_mean, projected_ground_truths_minus_mean):
        return {
            "coverage_risk": metrics.coverage_risk(svs, lower, upper, projected_ground_truths_minus_mean, lambda1=self.reconstruction_lambda, lambda2=self.coverage_lambda),
            "reconstruction_risk": metrics.reconstruction_risk(pcs, svs, ground_truths_minus_mean, projected_ground_truths_minus_mean, lambda1=self.reconstruction_lambda, pixel_ratio=self.opt.q),
            "interval_size": metrics.interval_size(lower, upper, svs, lambda1=self.reconstruction_lambda, lambda2=self.coverage_lambda, arch=self.archetype_matrix),
            "dimension": metrics.dimension(svs, lower, upper, lambda1=self.reconstruction_lambda, lambda2=self.coverage_lambda),
            "max_dimension": self.max_pcs,
            "uncertainty_volume": metrics.uncertainty_volume(lower, upper, svs, lambda1=self.reconstruction_lambda)
        }


"""
Acquisition strategies for active learning.

Implements various strategies for selecting which samples to label next:
- Random: Random sampling (passive learning baseline)
- Uncertainty: Select most uncertain predictions
- Diversity: Select diverse sequences
- Hybrid: Combine uncertainty and diversity
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm


class AcquisitionStrategy(ABC):
    """
    Base class for active learning acquisition strategies.
    
    An acquisition strategy decides which samples to acquire (label) next
    from the unlabeled pool, given the current model state.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize acquisition strategy.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
    
    @abstractmethod
    def select_batch(
        self,
        model: nn.Module,
        pool_dataset: Dataset,
        n_samples: int,
        device: torch.device,
        **kwargs
    ) -> np.ndarray:
        """
        Select a batch of samples from the pool.
        
        Args:
            model: Current trained model
            pool_dataset: Unlabeled pool dataset
            n_samples: Number of samples to select
            device: Device for computation
            **kwargs: Strategy-specific parameters
            
        Returns:
            Array of indices (relative to pool_dataset) to acquire
        """
        pass
    
    def get_name(self) -> str:
        """Get strategy name."""
        return self.__class__.__name__.replace('Acquisition', '').lower()


class RandomAcquisition(AcquisitionStrategy):
    """
    Random acquisition strategy (passive learning baseline).
    
    Randomly samples from the pool without using model information.
    This serves as the baseline that active learning strategies should beat.
    """
    
    def select_batch(
        self,
        model: nn.Module,
        pool_dataset: Dataset,
        n_samples: int,
        device: torch.device,
        **kwargs
    ) -> np.ndarray:
        """
        Randomly select samples from pool.
        
        Args:
            model: Not used for random selection
            pool_dataset: Unlabeled pool dataset
            n_samples: Number of samples to select
            device: Not used for random selection
            
        Returns:
            Array of randomly selected indices
        """
        pool_size = len(pool_dataset)
        
        if n_samples > pool_size:
            raise ValueError(
                f"Cannot select {n_samples} samples from pool of size {pool_size}"
            )
        
        # Random sampling without replacement
        indices = np.random.choice(pool_size, size=n_samples, replace=False)
        
        return indices


class UncertaintyAcquisition(AcquisitionStrategy):
    """
    Uncertainty-based acquisition using prediction variance.
    
    Uses an ensemble of models (or Monte Carlo dropout) to estimate
    prediction uncertainty, selecting samples where the model is most uncertain.
    
    For regression: Selects samples with highest prediction variance across ensemble.
    """
    
    def __init__(
        self,
        random_seed: Optional[int] = None,
        ensemble_models: Optional[List[nn.Module]] = None,
        mc_dropout_samples: int = 10,
        batch_size: int = 256
    ):
        """
        Initialize uncertainty acquisition.
        
        Args:
            random_seed: Random seed
            ensemble_models: List of models for ensemble uncertainty
            mc_dropout_samples: Number of forward passes for MC dropout
            batch_size: Batch size for inference
        """
        super().__init__(random_seed)
        self.ensemble_models = ensemble_models
        self.mc_dropout_samples = mc_dropout_samples
        self.batch_size = batch_size
    
    def select_batch(
        self,
        model: nn.Module,
        pool_dataset: Dataset,
        n_samples: int,
        device: torch.device,
        **kwargs
    ) -> np.ndarray:
        """
        Select samples with highest prediction uncertainty.
        
        If ensemble_models is provided, uses ensemble variance.
        Otherwise, uses MC dropout on the provided model.
        
        Args:
            model: Current trained model
            pool_dataset: Unlabeled pool dataset
            n_samples: Number of samples to select
            device: Device for computation
            
        Returns:
            Array of indices with highest uncertainty
        """
        pool_size = len(pool_dataset)
        
        if n_samples > pool_size:
            raise ValueError(
                f"Cannot select {n_samples} samples from pool of size {pool_size}"
            )
        
        # Get predictions and compute uncertainty
        if self.ensemble_models is not None:
            uncertainties = self._ensemble_uncertainty(
                self.ensemble_models, pool_dataset, device
            )
        else:
            uncertainties = self._mc_dropout_uncertainty(
                model, pool_dataset, device
            )
        
        # Select top-k most uncertain samples
        top_k_indices = np.argsort(uncertainties)[-n_samples:]
        
        return top_k_indices
    
    def _ensemble_uncertainty(
        self,
        models: List[nn.Module],
        dataset: Dataset,
        device: torch.device
    ) -> np.ndarray:
        """Compute uncertainty using ensemble variance."""
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Collect predictions from each model
        all_predictions = []
        
        for model in models:
            model.eval()
            model = model.to(device)
            
            predictions = []
            with torch.no_grad():
                for sequences, _ in tqdm(dataloader, desc="Ensemble prediction", leave=False):
                    sequences = sequences.to(device)
                    preds = model(sequences)
                    predictions.extend(preds.cpu().numpy())
            
            all_predictions.append(np.array(predictions))
        
        # Compute variance across ensemble
        all_predictions = np.array(all_predictions)  # shape: (n_models, n_samples)
        uncertainties = np.var(all_predictions, axis=0)
        
        return uncertainties
    
    def _mc_dropout_uncertainty(
        self,
        model: nn.Module,
        dataset: Dataset,
        device: torch.device
    ) -> np.ndarray:
        """Compute uncertainty using MC dropout."""
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        model.to(device)
        
        # Enable dropout during inference
        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()
        
        model.eval()
        model.apply(enable_dropout)
        
        # Collect predictions with dropout
        all_predictions = []
        
        for _ in range(self.mc_dropout_samples):
            predictions = []
            with torch.no_grad():
                for sequences, _ in dataloader:
                    sequences = sequences.to(device)
                    preds = model(sequences)
                    predictions.extend(preds.cpu().numpy())
            
            all_predictions.append(np.array(predictions))
        
        # Compute variance across MC samples
        all_predictions = np.array(all_predictions)  # shape: (n_mc_samples, n_samples)
        uncertainties = np.var(all_predictions, axis=0)
        
        return uncertainties


class DiversityAcquisition(AcquisitionStrategy):
    """
    Diversity-based acquisition using sequence embeddings.
    
    Selects diverse samples that provide good coverage of the sequence space,
    using model embeddings or k-mer representations.
    """
    
    def __init__(
        self,
        random_seed: Optional[int] = None,
        batch_size: int = 256,
        use_model_embeddings: bool = True,
        embedding_layer: Optional[str] = None
    ):
        """
        Initialize diversity acquisition.
        
        Args:
            random_seed: Random seed
            batch_size: Batch size for inference
            use_model_embeddings: Use model embeddings vs. k-mer features
            embedding_layer: Which layer to extract embeddings from
        """
        super().__init__(random_seed)
        self.batch_size = batch_size
        self.use_model_embeddings = use_model_embeddings
        self.embedding_layer = embedding_layer
    
    def select_batch(
        self,
        model: nn.Module,
        pool_dataset: Dataset,
        n_samples: int,
        device: torch.device,
        **kwargs
    ) -> np.ndarray:
        """
        Select diverse samples using k-center greedy algorithm.
        
        Args:
            model: Current trained model
            pool_dataset: Unlabeled pool dataset
            n_samples: Number of samples to select
            device: Device for computation
            
        Returns:
            Array of diverse sample indices
        """
        pool_size = len(pool_dataset)
        
        if n_samples > pool_size:
            raise ValueError(
                f"Cannot select {n_samples} samples from pool of size {pool_size}"
            )
        
        # Get embeddings for all pool samples
        if self.use_model_embeddings:
            embeddings = self._get_model_embeddings(model, pool_dataset, device)
        else:
            embeddings = self._get_kmer_embeddings(pool_dataset)
        
        # Use k-means clustering for diversity
        # Select cluster centers as diverse samples
        if n_samples < pool_size:
            kmeans = KMeans(n_clusters=n_samples, random_state=self.random_seed, n_init=10)
            kmeans.fit(embeddings)
            
            # Find nearest samples to cluster centers
            distances = euclidean_distances(kmeans.cluster_centers_, embeddings)
            indices = np.argmin(distances, axis=1)
        else:
            indices = np.arange(pool_size)
        
        return indices
    
    def _get_model_embeddings(
        self,
        model: nn.Module,
        dataset: Dataset,
        device: torch.device
    ) -> np.ndarray:
        """Extract embeddings from model's intermediate layer."""
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        model.eval()
        model.to(device)
        
        embeddings = []
        
        # Hook to capture embeddings
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        # Register hook (for now, use the output before final FC layer)
        # This assumes DREAMRNN structure - adjust for other models
        if hasattr(model, 'fc'):
            handle = model.fc.register_forward_hook(get_activation('fc_input'))
        
        with torch.no_grad():
            for sequences, _ in tqdm(dataloader, desc="Extracting embeddings", leave=False):
                sequences = sequences.to(device)
                _ = model(sequences)
                
                if 'fc_input' in activation:
                    # Use input to FC layer as embedding
                    emb = activation['fc_input'][0] if isinstance(activation['fc_input'], tuple) else activation['fc_input']
                    embeddings.append(emb.cpu().numpy())
                else:
                    # Fallback: use output
                    output = model(sequences)
                    embeddings.append(output.cpu().numpy())
        
        if hasattr(model, 'fc'):
            handle.remove()
        
        embeddings = np.vstack(embeddings)
        return embeddings
    
    def _get_kmer_embeddings(self, dataset: Dataset, k: int = 4) -> np.ndarray:
        """
        Create k-mer based embeddings for sequences.
        
        Simple fallback when model embeddings aren't available.
        """
        # This would require access to raw sequences
        # For now, just return random embeddings as placeholder
        # TODO: Implement proper k-mer counting
        embeddings = np.random.randn(len(dataset), 256)
        return embeddings


class HybridAcquisition(AcquisitionStrategy):
    """
    Hybrid acquisition combining uncertainty and diversity.
    
    Balances exploration (diversity) and exploitation (uncertainty) by
    selecting samples that are both uncertain and diverse.
    """
    
    def __init__(
        self,
        random_seed: Optional[int] = None,
        uncertainty_weight: float = 0.5,
        diversity_weight: float = 0.5,
        batch_size: int = 256,
        mc_dropout_samples: int = 10
    ):
        """
        Initialize hybrid acquisition.
        
        Args:
            random_seed: Random seed
            uncertainty_weight: Weight for uncertainty component
            diversity_weight: Weight for diversity component
            batch_size: Batch size for inference
            mc_dropout_samples: Number of MC dropout samples
        """
        super().__init__(random_seed)
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        
        # Create component strategies
        self.uncertainty_strategy = UncertaintyAcquisition(
            random_seed=random_seed,
            mc_dropout_samples=mc_dropout_samples,
            batch_size=batch_size
        )
        self.diversity_strategy = DiversityAcquisition(
            random_seed=random_seed,
            batch_size=batch_size
        )
    
    def select_batch(
        self,
        model: nn.Module,
        pool_dataset: Dataset,
        n_samples: int,
        device: torch.device,
        **kwargs
    ) -> np.ndarray:
        """
        Select samples by combining uncertainty and diversity scores.
        
        Args:
            model: Current trained model
            pool_dataset: Unlabeled pool dataset
            n_samples: Number of samples to select
            device: Device for computation
            
        Returns:
            Array of indices with highest combined score
        """
        pool_size = len(pool_dataset)
        
        if n_samples > pool_size:
            raise ValueError(
                f"Cannot select {n_samples} samples from pool of size {pool_size}"
            )
        
        # Get uncertainty scores
        uncertainties = self.uncertainty_strategy._mc_dropout_uncertainty(
            model, pool_dataset, device
        )
        # Normalize to [0, 1]
        uncertainties = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min() + 1e-8)
        
        # Get diversity scores (distance from already selected samples)
        # For simplicity, use negative distance to nearest selected sample
        # Here we'll use a simple heuristic: select greedily
        
        selected_indices = []
        remaining_indices = set(range(pool_size))
        
        # Get embeddings once
        embeddings = self.diversity_strategy._get_model_embeddings(
            model, pool_dataset, device
        )
        
        # Greedy selection
        for _ in range(n_samples):
            scores = np.zeros(pool_size)
            
            for idx in remaining_indices:
                # Uncertainty component
                uncertainty_score = uncertainties[idx]
                
                # Diversity component (distance to already selected)
                if len(selected_indices) > 0:
                    selected_embeddings = embeddings[selected_indices]
                    distances = np.linalg.norm(
                        embeddings[idx:idx+1] - selected_embeddings,
                        axis=1
                    )
                    diversity_score = np.min(distances)
                else:
                    diversity_score = 1.0  # First sample gets max diversity
                
                # Combined score
                scores[idx] = (
                    self.uncertainty_weight * uncertainty_score +
                    self.diversity_weight * diversity_score
                )
            
            # Select best
            best_idx = np.argmax(scores)
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return np.array(selected_indices)

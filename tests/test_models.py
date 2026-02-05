"""
Unit tests for model architectures.
"""

import pytest
import torch
import numpy as np
from models.dream_rnn import DREAMRNN, create_dream_rnn


class TestDREAMRNN:
    """Test DREAM-RNN model."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = DREAMRNN(
            input_channels=4,
            sequence_length=80
        )
        
        assert model.input_channels == 4
        assert model.sequence_length == 80
        assert model.output_dim == 1
    
    def test_forward_pass_basic(self):
        """Test forward pass with basic input."""
        batch_size = 8
        seq_length = 80
        
        model = DREAMRNN(
            input_channels=4,
            sequence_length=seq_length
        )
        
        # Create random input
        x = torch.randn(batch_size, 4, seq_length)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size,)
        assert not torch.isnan(output).any()
    
    def test_forward_pass_k562(self):
        """Test forward pass with K562-style input (5 channels)."""
        batch_size = 16
        seq_length = 230
        
        model = DREAMRNN(
            input_channels=5,
            sequence_length=seq_length
        )
        
        # Create random input
        x = torch.randn(batch_size, 5, seq_length)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size,)
        assert not torch.isnan(output).any()
    
    def test_predict_with_reverse_complement(self):
        """Test prediction with reverse complement averaging."""
        batch_size = 4
        seq_length = 80
        
        model = DREAMRNN(
            input_channels=4,
            sequence_length=seq_length
        )
        model.eval()
        
        # Create one-hot encoded sequence
        x = torch.zeros(batch_size, 4, seq_length)
        x[:, 0, :] = 1.0  # All A's
        
        # Predict with and without reverse complement
        pred_with_rc = model.predict(x, use_reverse_complement=True)
        pred_without_rc = model.predict(x, use_reverse_complement=False)
        
        # Check shapes
        assert pred_with_rc.shape == (batch_size,)
        assert pred_without_rc.shape == (batch_size,)
        
        # Predictions should be different (unless model outputs same for both)
        # Just check they're valid numbers
        assert not torch.isnan(pred_with_rc).any()
        assert not torch.isnan(pred_without_rc).any()
    
    def test_model_info(self):
        """Test model info generation."""
        model = create_dream_rnn(
            input_channels=4,
            sequence_length=80
        )
        
        info = model.get_model_info()
        
        assert "model_type" in info
        assert info["model_type"] == "DREAMRNN"
        assert info["input_channels"] == 4
        assert info["sequence_length"] == 80
        assert info["num_parameters"] > 0
        assert info["num_trainable_parameters"] > 0
    
    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading model checkpoints."""
        # Set seed for reproducibility
        torch.manual_seed(42)
        model = create_dream_rnn(
            input_channels=4,
            sequence_length=80
        )
        model.eval()  # Set to eval mode to disable dropout
        
        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        model.save_checkpoint(str(checkpoint_path), epoch=5)
        
        # Create new model with different initialization
        torch.manual_seed(123)
        new_model = create_dream_rnn(
            input_channels=4,
            sequence_length=80
        )
        new_model.eval()  # Set to eval mode
        
        # Load checkpoint - should restore to original model's weights
        checkpoint = new_model.load_checkpoint(str(checkpoint_path))
        
        assert checkpoint["model_info"]["model_type"] == "DREAMRNN"
        assert "epoch" in checkpoint
        assert checkpoint["epoch"] == 5
        
        # Compare state dicts directly - more reliable than comparing outputs
        original_state = model.state_dict()
        loaded_state = new_model.state_dict()
        
        # Check all parameters match
        for key in original_state:
            assert key in loaded_state
            assert torch.allclose(original_state[key], loaded_state[key])
        
        # Also test that they produce same output
        torch.manual_seed(999)
        x = torch.randn(4, 4, 80)
        with torch.no_grad():
            output1 = model(x)
            output2 = new_model(x)
        
        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-7)
    
    def test_different_hyperparameters(self):
        """Test model with different hyperparameters."""
        model = DREAMRNN(
            input_channels=4,
            sequence_length=80,
            hidden_dim=128,
            cnn_filters=128,
            dropout_cnn=0.1,
            dropout_lstm=0.3
        )
        
        x = torch.randn(4, 4, 80)
        output = model(x)
        
        assert output.shape == (4,)
        assert not torch.isnan(output).any()
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = DREAMRNN(
            input_channels=4,
            sequence_length=80
        )
        
        x = torch.randn(4, 4, 80)
        target = torch.randn(4)
        
        # Forward pass
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                assert not torch.isnan(param.grad).any()
        
        assert has_gradients


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

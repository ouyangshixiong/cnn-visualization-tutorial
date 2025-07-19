"""Tests for PyTorch Lightning models."""

import pytest
import torch
from src.models.pytorch import SimpleCNNClassifier, ResNetCNNClassifier


class TestSimpleCNNClassifier:
    """Tests for SimpleCNNClassifier."""
    
    @pytest.fixture
    def model(self):
        """Create a SimpleCNNClassifier instance."""
        return SimpleCNNClassifier(num_classes=10, learning_rate=1e-3)
    
    def test_model_creation(self, model):
        """Test model creation."""
        assert model is not None
        assert model.hparams.num_classes == 10
        assert model.hparams.learning_rate == 1e-3
    
    def test_forward_pass(self, model):
        """Test forward pass with CIFAR-10 input."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 32, 32)
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
    
    def test_training_step(self, model):
        """Test training step."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))
        
        loss = model.training_step((x, y), 0)
        assert isinstance(loss.item(), float)
        assert loss.item() > 0
    
    def test_validation_step(self, model):
        """Test validation step."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))
        
        model.validation_step((x, y), 0)
        # Should complete without errors


class TestResNetCNNClassifier:
    """Tests for ResNetCNNClassifier."""
    
    @pytest.fixture
    def model(self):
        """Create a ResNetCNNClassifier instance."""
        return ResNetCNNClassifier(
            num_classes=10, 
            learning_rate=1e-3, 
            pretrained=False,
            model_name="resnet18"
        )
    
    def test_model_creation(self, model):
        """Test model creation."""
        assert model is not None
        assert model.hparams.num_classes == 10
        assert model.hparams.model_name == "resnet18"
    
    def test_forward_pass(self, model):
        """Test forward pass with CIFAR-10 input."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 32, 32)
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
    
    def test_resnet50_model(self):
        """Test ResNet50 model creation."""
        model = ResNetCNNClassifier(
            num_classes=100,
            pretrained=False,
            model_name="resnet50"
        )
        
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 32, 32)
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 100)
    
    def test_invalid_model_name(self):
        """Test invalid model name raises error."""
        with pytest.raises(ValueError):
            ResNetCNNClassifier(model_name="invalid_model")
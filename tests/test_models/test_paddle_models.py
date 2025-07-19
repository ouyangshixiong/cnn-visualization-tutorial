"""Tests for PaddlePaddle models."""

import pytest
import paddle
from src.models.paddle import SimpleCNNClassifier, ResNetCNNClassifier


class TestSimpleCNNClassifier:
    """Tests for PaddlePaddle SimpleCNNClassifier."""
    
    @pytest.fixture
    def model(self):
        """Create a SimpleCNNClassifier instance."""
        return SimpleCNNClassifier(num_classes=10)
    
    def test_model_creation(self, model):
        """Test model creation."""
        assert model is not None
        assert hasattr(model, 'backbone')
    
    def test_forward_pass(self, model):
        """Test forward pass with CIFAR-10 input."""
        batch_size = 2
        input_tensor = paddle.randn([batch_size, 3, 32, 32])
        output = model(input_tensor)
        
        assert output.shape == [batch_size, 10]
        assert not paddle.isnan(output).any()


class TestResNetCNNClassifier:
    """Tests for PaddlePaddle ResNetCNNClassifier."""
    
    @pytest.fixture
    def model(self):
        """Create a ResNetCNNClassifier instance."""
        return ResNetCNNClassifier(
            num_classes=10,
            pretrained=False,
            model_name="resnet18"
        )
    
    def test_model_creation(self, model):
        """Test model creation."""
        assert model is not None
        assert hasattr(model, 'backbone')
    
    def test_forward_pass(self, model):
        """Test forward pass with CIFAR-10 input."""
        batch_size = 2
        input_tensor = paddle.randn([batch_size, 3, 32, 32])
        output = model(input_tensor)
        
        assert output.shape == [batch_size, 10]
        assert not paddle.isnan(output).any()
    
    def test_resnet50_model(self):
        """Test ResNet50 model creation."""
        model = ResNetCNNClassifier(
            num_classes=100,
            pretrained=False,
            model_name="resnet50"
        )
        
        batch_size = 2
        input_tensor = paddle.randn([batch_size, 3, 32, 32])
        output = model(input_tensor)
        
        assert output.shape == [batch_size, 100]
    
    def test_invalid_model_name(self):
        """Test invalid model name raises error."""
        with pytest.raises(ValueError):
            ResNetCNNClassifier(model_name="invalid_model")
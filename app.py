import numpy as np
import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import gc
import atexit
import os
import logging
from typing import Optional, Tuple
import psutil
import warnings
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter out specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.serialization')

# Device and memory configuration
MEMORY_THRESHOLD = 0.9  # 90% memory usage threshold
MAX_IMAGE_SIZE = 1024
BATCH_SIZE = 1

def get_device() -> torch.device:
    """
    Determine and configure the appropriate device for model execution.
    
    This function checks for CUDA (GPU) availability, then Apple Silicon MPS,
    and falls back to CPU if neither is available. It also configures device-specific
    settings for optimal performance.
    
    Returns:
        torch.device: The selected computation device (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_per_process_memory_fraction(0.7)
        logger.info("Using CUDA device")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    return device

# Initialize device at module level
device = get_device()

def clear_memory() -> None:
    """
    Clear memory based on device type.
    
    This function performs garbage collection and device-specific memory cleanup
    operations for CUDA and MPS devices.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        # Force MPS garbage collection
        torch.mps.empty_cache()

def check_memory() -> float:
    """
    Monitor system memory usage and trigger cleanup if necessary.
    
    Returns:
        float: Current memory usage percentage
    """
    memory = psutil.virtual_memory()
    if memory.percent > MEMORY_THRESHOLD * 100:
        logger.warning(f"High memory usage detected: {memory.percent}%")
        clear_memory()
    return memory.percent

norm_layer = nn.InstanceNorm2d

class ResidualBlock(nn.Module):
    """
    Residual block for the generator network.
    
    This block implements a residual connection with two convolutional layers,
    instance normalization, and ReLU activation.
    
    Args:
        in_features (int): Number of input channels
    """
    def __init__(self, in_features: int):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with residual connection
        """
        return x + self.conv_block(x)

class Generator(nn.Module):
    """
    Generator network for converting images to drawings.
    
    This network implements a U-Net like architecture with residual blocks
    and skip connections for better feature preservation.
    
    Args:
        input_nc (int): Number of input channels (typically 3 for RGB)
        output_nc (int): Number of output channels (typically 1 for grayscale drawings)
        n_residual_blocks (int, optional): Number of residual blocks. Defaults to 9.
        sigmoid (bool, optional): Whether to use sigmoid activation in the output. Defaults to True.
    """
    def __init__(self, input_nc: int, output_nc: int, n_residual_blocks: int = 9, sigmoid: bool = True):
        super(Generator, self).__init__()
        
        self.model0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            norm_layer(64),
            nn.ReLU(inplace=True)
        )

        # Downsampling with memory optimization
        in_features = 64
        model1 = []
        for _ in range(2):
            out_features = in_features * 2
            model1.extend([
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True)
            ])
            in_features = out_features
        self.model1 = nn.Sequential(*model1)

        # Residual blocks with memory optimization
        self.model2 = nn.ModuleList([ResidualBlock(in_features) for _ in range(n_residual_blocks)])

        # Upsampling with memory optimization
        model3 = []
        for _ in range(2):
            out_features = in_features // 2
            model3.extend([
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True)
            ])
            in_features = out_features
        self.model3 = nn.Sequential(*model3)

        model4 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7)
        ]
        if sigmoid:
            model4.append(nn.Sigmoid())
        self.model4 = nn.Sequential(*model4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_nc, height, width)
            
        Raises:
            MemoryError: If system memory usage is too high
        """
        if check_memory() > MEMORY_THRESHOLD * 100:
            raise MemoryError("System memory usage too high")
            
        x = self.model0(x)
        x = self.model1(x)
        for block in self.model2:
            x = block(x)
            clear_memory()
        x = self.model3(x)
        return self.model4(x)

class ModelManager:
    """
    Singleton class for managing model instances and their lifecycle.
    
    This class handles model loading, caching, and cleanup operations.
    It implements the singleton pattern to ensure only one instance exists.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.models = {}
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            self.initialized = True
            logger.info("ModelManager initialized")
            
    @lru_cache(maxsize=32)
    def get_model(self, style: str) -> Optional[nn.Module]:
        """
        Get a model instance for the specified style, loading it if necessary.
        
        Args:
            style (str): The style identifier for the model
            
        Returns:
            Optional[nn.Module]: The loaded model or None if not available
        """
        if style not in self.models:
            self.load_model(style)
        return self.models.get(style)
        
    def load_model(self, style: str) -> None:
        """
        Load a model for the specified style.
        
        Args:
            style (str): The style identifier for the model to load
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
            Exception: For other loading errors
        """
        try:
            model = Generator(3, 1, 3).to(device)
            model_path = os.path.join("models", f"{style}.pth")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file {model_path} not found")
            
            # Load state dict with device handling
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            
            self.models[style] = model
            logger.info(f"Successfully loaded model: {style} on {device}")
            
        except Exception as e:
            logger.error(f"Error loading model {style}: {str(e)}")
            raise
            
    def cleanup(self) -> None:
        """
        Clean up all loaded models and free memory.
        
        This method ensures proper resource cleanup by moving models to CPU
        and clearing GPU memory if applicable.
        """
        try:
            for model_name in list(self.models.keys()):
                if self.models[model_name] is not None:
                    self.models[model_name].cpu()
                    del self.models[model_name]
            self.models.clear()
            clear_memory()
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

def process_image(input_img: str) -> Image.Image:
    """
    Process and optimize an input image for model inference.
    
    Args:
        input_img (str): Path to the input image file
        
    Returns:
        Image.Image: Processed PIL Image
        
    Raises:
        Exception: If image processing fails
    """
    try:
        with Image.open(input_img) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Optimize image size
            if max(img.size) > MAX_IMAGE_SIZE:
                ratio = MAX_IMAGE_SIZE / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
            # Return a copy of the processed image
            return img.copy()
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

model_manager = ModelManager()

def predict(input_img: str, ver: str) -> Image.Image:
    """
    Generate a drawing from an input image using the specified style.
    
    Args:
        input_img (str): Path to the input image file
        ver (str): Style version to use ('anime', 'contour', or 'sketch')
        
    Returns:
        Image.Image: Generated drawing as a PIL Image
        
    Raises:
        gr.Error: If prediction fails
    """
    try:
        check_memory()
        
        # Get model (cached)
        model = model_manager.get_model(ver)
        if model is None:
            raise ValueError(f"Model {ver} not available")
            
        # Process image
        input_img = process_image(input_img)
        input_tensor = model_manager.transform(input_img).unsqueeze(0).to(device)
        
        # Generate drawing
        with torch.no_grad():
            drawing = model(input_tensor)[0]
            # Move result to CPU before detaching
            if device.type in ['cuda', 'mps']:
                drawing = drawing.cpu()
            drawing = drawing.detach()
            
        # Convert to PIL Image
        result = transforms.ToPILImage()(drawing)
        
        # Cleanup
        del input_tensor, drawing
        clear_memory()
        
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise gr.Error(f"An error occurred: {str(e)}")

# Register cleanup
atexit.register(model_manager.cleanup)

def main() -> None:
    """
    Main entry point for the Gradio interface.
    
    This function sets up the Gradio interface configuration, initializes
    the model manager and launches the interface. It also handles any
    exceptions that may occur during interface launch and cleanup.
    
    Returns:
        None
    """
    try:
        # Initialize model manager
        model_manager.load_model('anime')
        model_manager.load_model('contour')
        model_manager.load_model('sketch')

        # Gradio interface configuration
        title = "Informative Drawings"
        description = f"Generate line drawings from images using three different styles. Running on: {device}"

        iface = gr.Interface(
            fn=predict,
            inputs=[
                gr.Image(type='filepath'),
                gr.Radio(['anime', 'contour', 'sketch'], type="value", label='Style')
            ],
            outputs=gr.Image(type="pil", format="png"),
            title=title,
            description=description,
            flagging_mode="never",
        )

        try:
            iface.launch(
                show_error=True,
                share=False,
                server_port=None  # Let Gradio choose an available port
            )
        except Exception as e:
            logger.error(f"Error launching interface: {str(e)}")
        finally:
            model_manager.cleanup()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()

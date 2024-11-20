import numpy as np
import torch
import torch.nn as nn
import streamlit as st
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
import io

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

# Custom CSS to improve layout
CUSTOM_CSS = """
<style>
    /* Modern color scheme and base styles */
    :root {
        --primary-color: #4F46E5;
        --secondary-color: #7C3AED;
        --border-radius: 0.5rem;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* Light mode colors */
    [data-theme="light"] {
        --background-color: #F9FAFB;
        --text-color: #1F2937;
        --surface-color: #FFFFFF;
        --border-color: #E5E7EB;
        --hover-color: #F3F4F6;
    }

    /* Dark mode colors */
    [data-theme="dark"] {
        --background-color: #111827;
        --text-color: #F9FAFB;
        --surface-color: #1F2937;
        --border-color: #374151;
        --hover-color: #374151;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
    }

    /* Global styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Container styles */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
        background: var(--surface-color);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
    }

    /* Header styles */
    .stTitle {
        color: var(--text-color);
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 1rem;
    }

    /* Section headers */
    [data-testid="stMarkdown"] h3 {
        margin-top: 0;
        margin-bottom: 1rem;
        color: var(--text-color);
    }

    /* Column layout */
    [data-testid="column"] {
        background: var(--surface-color);
        padding: 1rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        margin: 0.5rem;
        display: flex !important;
        flex-direction: column !important;
    }

    /* File uploader */
    .stFileUploader {
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: var(--border-radius);
        border: 2px dashed var(--border-color);
        background: var(--surface-color);
    }

    .stFileUploader:hover {
        border-color: var(--primary-color);
        background: var(--hover-color);
    }

    /* Radio buttons */
    .stRadio {
        margin: 1rem 0;
    }

    .stRadio > div {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .stRadio label {
        background: var(--surface-color);
        padding: 0.5rem 1rem;
        border-radius: var(--border-radius);
        border: 1px solid var(--border-color);
        cursor: pointer;
        transition: all 0.2s;
        color: var(--text-color);
    }

    .stRadio label:hover {
        border-color: var(--primary-color);
        background: var(--hover-color);
    }

    /* Button styles */
    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: var(--border-radius);
        font-weight: 500;
        transition: transform 0.2s, box-shadow 0.2s;
        margin-top: 0.5rem;
        width: 100%;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow);
    }

    /* Dark mode specific overrides */
    [data-theme="dark"] .stButton > button,
    [data-theme="dark"] .stDownloadButton > button {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }

    /* Images */
    .element-container div[data-testid="stImage"] {
        max-height: 50vh;
        overflow: hidden;
    }

    .element-container div[data-testid="stImage"] img {
        max-height: 50vh;
        width: 100%;
        object-fit: contain;
        border-radius: var(--border-radius);
    }

    /* Info messages */
    .stAlert {
        background: var(--surface-color);
        border-radius: var(--border-radius);
        border-left: 4px solid var(--primary-color);
        padding: 1rem;
        margin: 1rem 0;
        color: var(--text-color);
    }

    /* Loading spinner */
    .stSpinner > div {
        border-color: var(--primary-color);
    }

    /* Description text */
    .stMarkdown {
        color: var(--text-color);
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    /* Image captions */
    .stImage caption {
        color: var(--text-color);
    }

    /* Dark mode specific overrides */
    [data-theme="dark"] .stButton > button,
    [data-theme="dark"] .stDownloadButton > button {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }

    [data-theme="dark"] .stFileUploader {
        border-color: var(--border-color);
    }
</style>
"""

def get_device() -> torch.device:
    """
    Determine and configure the appropriate device for model execution.
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
    """Clear memory based on device type."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        # Force MPS garbage collection
        torch.mps.empty_cache()

def check_memory() -> float:
    """Monitor system memory usage and trigger cleanup if necessary."""
    memory = psutil.virtual_memory()
    if memory.percent > MEMORY_THRESHOLD * 100:
        logger.warning(f"High memory usage detected: {memory.percent}%")
        clear_memory()
    return memory.percent

norm_layer = nn.InstanceNorm2d

class ResidualBlock(nn.Module):
    """Residual block for the generator network."""
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
        return x + self.conv_block(x)

class Generator(nn.Module):
    """Generator network for converting images to drawings."""
    def __init__(self, input_nc: int, output_nc: int, n_residual_blocks: int = 9, sigmoid: bool = True):
        super(Generator, self).__init__()
        
        self.model0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            norm_layer(64),
            nn.ReLU(inplace=True)
        )

        # Downsampling
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

        # Residual blocks
        self.model2 = nn.ModuleList([ResidualBlock(in_features) for _ in range(n_residual_blocks)])

        # Upsampling
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
    """Singleton class for managing model instances and their lifecycle."""
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
        """Get a model instance for the specified style."""
        if style not in self.models:
            self.load_model(style)
        return self.models.get(style)
        
    def load_model(self, style: str) -> None:
        """Load a model for the specified style."""
        try:
            model = Generator(3, 1, 3).to(device)
            model_path = os.path.join("models", f"{style}.pth")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file {model_path} not found")
            
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
        """Clean up all loaded models and free memory."""
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

def process_image(input_img: Image.Image) -> Image.Image:
    """Process and optimize an input image for model inference."""
    try:
        # Convert to RGB if necessary
        if input_img.mode != 'RGB':
            input_img = input_img.convert('RGB')
            
        # Optimize image size
        if max(input_img.size) > MAX_IMAGE_SIZE:
            ratio = MAX_IMAGE_SIZE / max(input_img.size)
            new_size = tuple(int(dim * ratio) for dim in input_img.size)
            input_img = input_img.resize(new_size, Image.Resampling.LANCZOS)
            
        return input_img
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

model_manager = ModelManager()

def predict(input_img: Image.Image, style: str) -> Image.Image:
    """Generate a drawing from an input image using the specified style."""
    try:
        check_memory()
        
        # Get model (cached)
        model = model_manager.get_model(style)
        if model is None:
            raise ValueError(f"Model {style} not available")
            
        # Process image
        input_img = process_image(input_img)
        input_tensor = model_manager.transform(input_img).unsqueeze(0).to(device)
        
        # Generate drawing
        with torch.no_grad():
            drawing = model(input_tensor)[0]
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
        st.error(f"An error occurred: {str(e)}")
        return None

# Register cleanup
atexit.register(model_manager.cleanup)

def main():
    """
    Main entry point for the Streamlit interface.
    """
    try:
        st.set_page_config(
            page_title="Informative Drawings",
            page_icon="✏️",
            layout="centered",
            initial_sidebar_state="collapsed",
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': None
            }
        )
        
        # Inject custom CSS
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        
        # Main title and description
        st.title("✏️ Informative Drawings")
        st.markdown(f"""
        Transform your images into beautiful line drawings using AI. 
        Currently running on: **{device}**
        """)
        
        # Initialize models
        with st.spinner("Loading models..."):
            for style in ['anime', 'contour', 'sketch']:
                try:
                    model_manager.load_model(style)
                except Exception as e:
                    st.error(f"Failed to load {style} model: {str(e)}")
                    return

        # Create main containers
        input_col, output_col = st.columns([1, 1])
        
        # Input section
        with input_col:
            st.markdown("### 📸 Input Image")
            uploaded_file = st.file_uploader(
                "Choose an image to transform",
                type=["jpg", "jpeg", "png"],
                help="Upload a JPG or PNG image"
            )
            
            # Style selection
            style = st.radio(
                "🎨 Select Drawing Style",
                ['anime', 'contour', 'sketch'],
                horizontal=True,
                help="Choose the style for your drawing"
            )
            
            if uploaded_file is not None:
                try:
                    input_image = Image.open(uploaded_file)
                    st.image(
                        input_image,
                        caption="Original Image",
                        use_column_width=True,
                        clamp=True
                    )
                    
                    # Generate button
                    if st.button("🎨 Generate Drawing", type="primary", use_container_width=True):
                        # Output section
                        with output_col:
                            st.markdown("### 🖼️ Generated Drawing")
                            with st.spinner("🎨 Creating your drawing..."):
                                result = predict(input_image, style)
                                if result is not None:
                                    st.image(
                                        result,
                                        caption=f"{style.title()} Style Drawing",
                                        use_column_width=True,
                                        clamp=True
                                    )
                                    
                                    # Download button
                                    buf = io.BytesIO()
                                    result.save(buf, format='PNG')
                                    st.download_button(
                                        label="⬇️ Download Drawing",
                                        data=buf.getvalue(),
                                        file_name=f"drawing_{style}.png",
                                        mime="image/png",
                                        use_container_width=True
                                    )
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
            else:
                with output_col:
                    st.markdown("### 🖼️ Generated Drawing")
                    st.info("Upload an image to get started!")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()

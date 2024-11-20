# Informative Drawings App

A Gradio web application that transforms photographs into beautiful line drawings using AI. This app is based on the research paper ["Learning to generate line drawings that convey geometry and semantics"](https://arxiv.org/abs/2203.12691) by Chan et al.

![Informative Drawings Demo](https://carolineec.github.io/informative_drawings/images/teaser.png)

## Features 

-  Three unique drawing styles: anime, contour, and sketch
-  Fast processing with GPU/MPS acceleration when available
-  Modern, intuitive Gradio interface
-  Responsive design that works on desktop and mobile
-  Memory-optimized for efficient processing

## Installation

1. Clone this repository:
```bash
git clone https://github.com/jarettrude/informative-drawings.git
cd informative-drawings
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

3. Install dependencies:
```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

## Usage 

1. Start the application:
```bash
python app.py
```

2. Open your web browser to the URL shown in the terminal (typically http://localhost:7860)

3. Upload an image and select your preferred drawing style

4. Click "Submit" to generate your drawing

5. Download the result using the "Download Drawing" button

## Requirements 

- Python 3.12.7
- PyTorch 2.5.1
- Gradio 5.6.0
- Other dependencies listed in `requirements.txt`

## Credits 

This application is built upon the research work:

```bibtex
@article{chan2022drawings,
    title={Learning to generate line drawings that convey geometry and semantics},
    author={Chan, Caroline and Durand, Fredo and Isola, Phillip},
    booktitle={CVPR},
    year={2022}
}
```

- Original Project: [carolineec/informative-drawings](https://github.com/carolineec/informative-drawings)
- [Project Page](https://carolineec.github.io/informative_drawings/)
- [Research Paper](https://arxiv.org/abs/2203.12691)

## License 

This project is licensed under the terms of the original informative-drawings repository license.
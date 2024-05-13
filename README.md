# Confluence

This repo contains the code for the project 'Confluence'. 

It contains the code for preprocessing, negative sampling, user and item profile generation and the two tower model implementation.

# Steps to Run code

1. Create a virtual environment
2. Install dependencies using `pip install -r requirements.txt`
3. Installing  PyTorch dependencies
    - On local: Just use `pip install torch`
    - On HPRC:
        1. `pip install --upgrade pip`
        2. `pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html`

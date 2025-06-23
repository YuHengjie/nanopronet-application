# nanopronet-application

## Model Deployment Process Guide

This guide outlines the steps to set up the necessary environment and deploy NanoProFormer. This guide provides the specific versions used in this study for reproducibility (this study is based on LINUX and NVIDIA L40 and NVIDIA L40S GPUs). Non-uniform versions should also be able to use the model.

### 1. Environment Creation with Conda

First, we will create a new Conda environment named **nanoproformer**. This isolates our project dependencies from other Python projects on your system.

```bash
conda create -n nanoproformer python==3.11.3  # You can choose a different Python version if required
conda activate nanoproformer
```

### 2. PyTorch Installation (OS and CUDA Version Specific)
The installation of PyTorch depends on your operating system (OS) and the CUDA version you have installed (if you plan to use a GPU). Please check pytorch website (https://pytorch.org/get-started/locally/) to obtain required command.

For CUDA 12.1 (Linux, GPU support):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Installing Other Dependencies from requirements.txt
After installing PyTorch, you can install the rest of the required packages listed in the requirements.txt file. We will specifically exclude PyTorch itself, as it has already been installed.

```bash
pip install -r requirements.txt
```

### 4. Loading Code and Model Parameters
Next, you need to download the project code and navigate into its directory.

```bash
git clone https://github.com/YuHengjie/nanopronet-application.git
cd nanopronet-application
```

### 5. Downloading Pre-trained Models

You will need to download the pre-trained models, including esm2_t36_3B_UR50D and Linq-Embed-Mistral, These models should be placed in specific directories (/pretrained_model/esm2_t36_3B_UR50D and /pretrained_model/Linq-Embed-Mistral) to ensure proper access by the application.

### 6. Execute Zero-shot Inference Programs
First, place your raw data files into a /data directory within the main project directory. This will be the input for the subsequent processing steps.

Navigate to the /protein_embedding directory and run the necessary script to encode your protein sequences. This step transforms raw protein sequences into a numerical representation that the model can understand.

Next, move to the /X_embedding directory and execute the script for structured text encoding.

After encoding your input features, execute the data_process.py script located in the main project folder. This script handles the necessary label processing for your dataset.

Finally, run the zero-shot inference programs. You'll typically use either model_unseen_binary.py for binary classification tasks or model_unseen_reg.py for regression tasks, depending on your model's objective.

After these steps, your zero-shot inference results will be generated based on the processed data. 

# Phi-3 Fine-tuning with QLoRA and Azure Machine Learning

This project demonstrates how to fine-tune Microsoft's Phi-3 language model using QLoRA (Quantization-aware Low-Rank Adapter Tuning for Language Generation) with the help of Azure Machine Learning. QLoRA allows efficient fine-tuning of large language models by quantizing a pre-trained model to 4-bit precision while attaching small "Low-Rank Adapters" to the model.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The aim of this project is to fine-tune the **Phi-3 mini** language model using QLoRA on a sample dataset from Hugging Face (UltraChat 200k). Fine-tuning is done with Azure Machine Learning resources, leveraging GPU-powered compute clusters. This project showcases efficient large language model fine-tuning for specific tasks using minimal computational resources.

## Features
- **Efficient Fine-Tuning**: Fine-tunes Phi-3 mini with QLoRA by attaching low-rank adapters to a quantized model.
- **Azure Integration**: Utilize Azure ML SDK to create compute clusters and manage fine-tuning jobs.
- **Customizable**: Easily adjust LoRA configuration, dataset, and training parameters.
- **Dataset Preparation**: Handles dataset loading, splitting, and saving to JSON format for fine-tuning.
- **Scalability**: Configured to scale with available compute resources like A100 GPUs.

## Setup
Follow these steps to set up the project locally.

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Set Up the Virtual Environment
Create a Python virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # For Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
Install the required Python packages using `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Azure Setup
- Make sure you have an **Azure subscription** and have created a **machine learning workspace**.
- Install the Azure ML SDK and log in:
    ```bash
    pip install azure-ai-ml azure-identity
    ```

## Usage
### Running the Fine-Tuning Pipeline

1. **Prepare the Dataset**: Modify the dataset configuration in `src/data_prep.py` if necessary.
2. **Fine-Tuning**: Run the fine-tuning script using:

    ```bash
    python -m src.train
    ```

This will:
- Load the UltraChat dataset from Hugging Face.
- Fine-tune the **Phi-3 mini** model using QLoRA with low-rank adapters.
- Use the GPU for training if available (CUDA enabled).

### Customizing Parameters
You can modify parameters such as learning rate, batch size, and dataset splits in `src/config.py`. Example:

```python
# src/config.py
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TEST_SPLIT_SIZE = 0.2
```

## Project Structure

```bash
.
├── src/
│   ├── __init__.py             # Initializes the src package
│   ├── data_prep.py            # Dataset preparation (load, split, save)
│   ├── model_setup.py          # Model and LoRA configuration setup
│   ├── training.py             # Main training loop
│   ├── config.py               # Configuration file for hyperparameters
├── environment/
│   └── conda.yml               # Conda environment configuration (optional)
├── train.py                    # Entry point for the training script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Ignored files and folders
└── README.md                   # This file
```

## Contributing
Contributions are welcome! Please follow these steps to contribute to the project:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/my-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add some feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/my-feature
   ```
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

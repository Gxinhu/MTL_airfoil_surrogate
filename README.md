<div align="center">

# MTL-Airfoil-Surrogate: Multi-Task Learning for Enhanced Airfoil Design

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![Config: Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc/)
[![Template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/ashleve/lightning-hydra-template)
[![Paper](https://img.shields.io/badge/Paper-POF-blue)](https://doi.org/10.1063/5.0258928)


</div>

**Official implementation for the paper: "Enhancing Airfoil Design Optimization Surrogate Models Using Multi-Task Learning: Separating Airfoil Surface and Fluid Domain Predictions"**

**Boost Airfoil Design with Multi-Task Learning.** This repository provides the code for enhancing airfoil design optimization using Multi-Task Learning (MTL). Our approach improves surrogate model accuracy by separately predicting airfoil surface characteristics and fluid domain behavior, leading to faster airfoil design optimization.

## ✨ Key Features

* **Multi-Task Learning (MTL) Approach:** Improves surrogate model accuracy by decoupling airfoil surface characteristics (e.g., pressure distribution) and fluid domain predictions (e.g., lift, drag coefficients).
* **Reproducible Research:** Provides all code, configurations, and instructions to reproduce our experimental results.
* **Powered by PyTorch Lightning & Hydra:** Ensures clean, organized, and configurable experiments.
* **Fast Environment Setup with `uv` (Recommended):** Leverage modern tooling for efficient dependency management.

## 🚀 Getting Started

Follow these steps to get started with training and experimenting with our MTL-Airfoil-Surrogate models.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Gxinhu/MTL_airfoil_surrogate.git
   cd MTL_airfoil_surrogate
   ```

2. **Environment Setup:** Choose your preferred method for setting up the Python environment.

   * **Option 1: Recommended - Fast and Modern with `uv`**

     ```bash
     uv sync  # Create and synchronize a virtual environment using uv
     source .venv/bin/activate  # Activate the virtual environment (Linux/macOS)
     .venv\Scripts\activate  # Activate the virtual environment (Windows)
     ```

     **Note:** `uv` is a significantly faster package installer and resolver. Ensure you have it installed by following the instructions on [uv installation guide](https://astral.sh/uv).

   * **Option 2: Using `pip` (Standard)**

     ```bash
     pip install -r requirements.txt
     ```

3. **Dataset Preparation:**

   * **Download the AirfRANS dataset:** Obtain the dataset from [AirfRANS dataset documentation](https://airfrans.readthedocs.io/en/latest/notes/dataset.html). Download and extract it to a location on your system.

   **Note:** The AirfRANS dataset is approximately 9.3GB in size.

   * **Configure Dataset Path:**
     * Create a `.env` file in the root directory of the repository.
     * Add the following line to `.env`, replacing `/path/to/your/dataset` with the *absolute path* to your AirfRANS dataset directory:

       ```env
       DATASET=/path/to/your/dataset
       ```

       **Example:** If your dataset is in `/Users/yourusername/datasets/airfoil_data`, your `.env` file should contain:

       ```env
       DATASET=/Users/yourusername/datasets/airfoil_data
       ```

4. **Training the Model:**

   This repository allows you to train both Baseline models and Multi-Task Learning (MTL) models.  The available model architectures and options differ slightly between these two approaches.

   * **Baseline Model Training:**

     ```bash
     python src/train.py experiment=airflow model=mlp model.is_whole=True logger=csv
     # Trains a baseline model using an MLP backbone and global loss.
     ```

     * `experiment=airflow`:  Selects the configuration for baseline airflow prediction.
     * `model=mlp`:  Specifies the **MLP backbone architecture** for the Baseline model.
     * `model.is_whole=True`:  `True` is `global loss`, `False` is `composite loss`
     * `logger=csv`: experiment tracking by a csv logger

     **Available Baseline Model Architectures (`model` parameter for Baseline):**
        * `mlp`: Multi-Layer Perceptron backbone.
        * `pointnet`: PointNet backbone.

   * **MTL Model Training:**

     ```bash
     python src/train.py experiment=airflow_mtl model=mtl_mlp_I model.loss_term=stch logger=csv
     # Trains an MTL model using MTL-MLP-I architecture and STCH optimization.
     ```

     * `experiment=airflow_mtl`: Selects the configuration for MTL-based airflow prediction.
     * `model=mtl_mlp_I`:  Specifies the **MTL-MLP-I architecture** for the MTL model.
     * `model.loss_term=stch`:  Chooses the **STCH MTL optimization strategy**.

     **Available MTL Model Architectures (`model` parameter for MTL):**
        * `mtl_mlp_I`: MTL model with MLP backbone, Decoder I configuration.
        * `mtl_mlp_II`: MTL model with MLP backbone, Decoder II configuration.
        * `mtl_point_I`: MTL model with PointNet backbone, Decoder I configuration.
        * `mtl_point_II`: MTL model with PointNet backbone, Decoder II configuration.

     **Available MTL Optimization Strategies (`model.loss_term` parameter for MTL):**
        * `stch`: STCH optimization strategy.
        * `fairgrad`: FairGrad optimization strategy.
        * `famo`: FAMO optimization strategy.

   * **Explore Configurations:**  Refer to the `configs/` directory and [lightning-hydra-template)](https://github.com/ashleve/lightning-hydra-template) for more example configuration files. You can adjust these files to experiment with different models, MTL strategies (for MTL models), and hyperparameters.

## ✍️ Citation

If you use this codebase in your research, please cite our paper:

```bibtex
@article{10.1063/5.0258928,
    author = {Hu, Xin and An, Bo and Guan, Yongke and Li, Dong and Mellibovsky, Fernando and Sang, Weimin and Wang, Gang},
    title = {Enhancing airfoil design optimization surrogate models using multi-task learning: Separating airfoil surface and fluid domain predictions},
    journal = {Physics of Fluids},
    volume = {37},
    number = {3},
    pages = {037175},
    year = {2025},
    month = {03},
    issn = {1070-6631},
    doi = {10.1063/5.0258928},
    url = {https://doi.org/10.1063/5.0258928},
    eprint = {https://pubs.aip.org/aip/pof/article-pdf/doi/10.1063/5.0258928/20447266/037175\_1\_5.0258928.pdf},
}

```

## 📜 License

This project is licensed under the [MIT License](LICENSE) - see the `LICENSE` file for details.

## References

[STCH](https://github.com/Xi-L/STCH)

[FairGrad](https://github.com/OptMN-Lab/fairgrad)

[FAMO](https://github.com/Cranial-XIX/FAMO)

[AirfRANS](https://github.com/Extrality/AirfRANS)

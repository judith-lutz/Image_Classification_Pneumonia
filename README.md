# Image classification with medical X-Ray images 💊🏥

> We use a Kaggle dataset of X-Ray images to determine whethter the image belongs to a healthy person or not. We use pytorch to build anc evaluate different models. 

## 📊 Project overview

**Problem:** 
The data consist of X-Ray images of the upper part of the body. 

**Objective:** 
We try to predict whether an image shows a healthy/normal person or a person with pneumonia. 

**Methods:** 
We us PyTorch. 



## Setup

- Clone the repository
```bash
# Clone repository 
git clone https://github.com/judith-lutz/Image_Classification_Pneumonia
cd Image_Classification_Pneumonia
```

- Install [uv](https://uv.dev) (if not installed already) and synchronise dependencies
```bash
# Install dependencies 
uv sync
```
- Install [PyTorch](https://pytorch.org/) (if not installed already).

### Execution

Run the notebooks in the following order:
1. notebooks/01_exploration.ipynb
2. notebooks/02_preprocessing.ipynb
3. notebooks/03_modeling.ipynb
<!--
4. notebooks/04_results.ipynb
-->


### Origin of the data

1. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning
Kermany, Daniel S. et al.
Cell, Volume 172, Issue 5, 1122 - 1131.e9

2. Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, V2, doi: 10.17632/rscbjbr9sj.2

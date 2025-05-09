# 🫀 3D Heart Segmentation
> **What if we could transform raw MRI scans into interactive 3D models, revealing critical cardiac structures at a glance?**

![small_heart_image](https://github.com/user-attachments/assets/8720c7a3-a2ca-4d38-be1d-0983ba7e6907)

## Overview
This project explores the **intersection of medical imaging and computer vision**, focusing on **transforming volumetric MRI data into detailed 3D heart models** while preparing refined data for a **deep learning model** that predicts accurate segmentation masks. By applying advanced image processing techniques, correcting intensity distortions, and enhancing 3D mesh reconstructions, the system improves both **segmentation accuracy and diagnostic insights**.

#### 🎯 **The ultimate goal is to improve medical analysis and diagnostic workflows, by leveraging 3D visualization and automated segmentation of critical heart structures.**

## Project Objectives
To achieve this, the project focuses on:
- **Generating 3D heart models** through surface mesh reconstruction.  
- **Training a predictive deep learning model** to segment heart structures based on refined MRI data and patient conditions.  

## **What This Project Covers**
- **Preprocessing & Standardization:** Correcting intensity distortions and artifacts through metric-based analysis.  
- **3D Reconstruction & Refinement:** Using the **Ball-Pivoting Algorithm (BPA)** with **Gaussian smoothing** and thresholding to generate an anatomically accurate 3D heart mesh.  
- **Data & Metric Analysis:** Investigating correlations between **patient metadata** and MRI scan metrics to guide model learning.  
- **Segmentation Model Preparation:** Preparing clean, high-quality input data for a deep learning model that predicts segmentation masks based on patient conditions.  

## Technologies Used
- **Programming Language:** Python  
- **3D Reconstruction Library:** Open3D (for BPA mesh processing and surface smoothing)  
- **Image Processing:** NumPy, SciPy, and OpenCV  
- **Visualization:** Matplotlib and Plotly  
- **Data Analysis:** Pandas and Seaborn  
- **Deep Learning:** 3D UNet and MONAI.

## Repository Structure
- `metrics_and_processing.ipynb` – Analyzing scan metrics, identifying correlations, and evaluating processing techniques to refine preprocessing.
- `src/preprocessing.py` – Preprocessing volumetric MRI data, correcting distortions, and preparing standardized inputs as a modular pipeline.  
- `3d_reconstruction.ipynb` - Visualizing 3D heart models and their segmentation.
- `segmentation_model.ipynb` - Model training for predicting accurate masks based on refined data. 

## 🚀 Project Outcomes
Upon completing this project, the system successfully:  
- **Transformed raw MRI scans** into accurate 3D heart models, enhancing anatomical visualization.  
- **Standardized volumetric MRI data** by addressing artifacts and intensity distortions.  
- **Developed a segmentation model** to predict masks based on patient conditions, utilizing refined MRI data and 3D heart models.  

## Dataset
 Dataset from ACDC challenge:
  https://humanheart-project.creatis.insa-lyon.fr/database/

*O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al.
"Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved ?" in IEEE Transactions on Medical Imaging, vol. 37, no. 11, pp. 2514-2525, Nov. 2018
doi: 10.1109/TMI.2018.2837502*

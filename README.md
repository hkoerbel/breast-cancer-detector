# 🧬 Breast Cancer Detection – Machine Learning Pipeline

This repository contains the full workflow for building, evaluating, and comparing machine-learning models for binary breast cancer detection based on structured medical data.
The project was developed as part of a data-processing and machine-learning assignment and demonstrates a complete ML lifecycle — from preprocessing to model benchmarking and interpretation.

-----

# 💡 Suggestion

The whole project is documented in detail in the attached notebook. For more information than in the README please check the notebook.

-----

# 📌 Objectives

* Build a reproducible data-processing and modeling pipeline for cancer classification.
* Finetune two different base models and compare their performance.
* Evaluate performance using standardized metrics such as accuracy, precision and recall.
  
-----

# 🛢️ Data

* Dataset 1: RSNA Screening Mammography
  
  Used for: baseline inference, fine tuning, evaluation

  https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-256-pngs

* Dataset 2: KAU-BCMD Mammography
  
  Used for: zero-shot inference, fine tuning, evaluation

  https://www.kaggle.com/datasets/orvile/kau-bcmd-mamography-dataset
  
-----

# 🧠 Base Models Used

# Model 1

EfficientNet-B0 Breast Cancer Classification (Hugging Face)

https://huggingface.co/keanteng/efficientnet-b0-breast-cancer-classification-0604-2

Key characteristics:
* EfficientNet-B0 architecture
* Pretrained on ImageNet
* Fine tuned on the Mini-DDBS-JPEG mammography dataset
* Input size: 256×256
* Binary labels: Has_Cancer (1) vs Normal (0)

# Model 2

ResNet-50 (Torchvision)

https://huggingface.co/microsoft/resnet-50

Key characteristics:
* ResNet-50 architecture with residual connections
* Pretrained on ImageNet
* Widely used as a strong and stable baseline for medical imaging
* Default input size 224×224 (we upscale to 256×256 for consistency)
* Binary classification head adapted for Cancer vs No Cancer prediction
  
-----

# 📊 Results

* Recognizing tumors works well.
* Differentiating benign tumours from malignant tumours is something the models struggle with.
* Performance on the RSNA dataset always lacks behind performance on the KAU dataset, because in the RSNA dataset, benign tumours are classified as non-cancerous, confusing the models. For the KAU dataset we could decide on our own, to classify benign and malignant tumours both as cancerous
* The best model is the ResNet-50 Model trained on KAU data. 

<img width="900" height="449" alt="grafik" src="https://github.com/user-attachments/assets/5cea72e5-8971-4875-adcd-4ed1ed89a3a9" />
<img width="900" height="459" alt="grafik" src="https://github.com/user-attachments/assets/edbb446d-8a4e-45bf-a452-11358d10239f" />

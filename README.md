# NeuraScan - Tumor Detector
# Project Link: https://tumor-detector-system.onrender.com/

##  Overview

This project is a **deep learning-based Brain Tumor Detection System** that leverages **Convolutional Neural Networks (CNNs)** with **VGG16 architecture** to classify MRI brain scans into **tumor** and **non-tumor** categories.

The system is trained on MRI image datasets and deployed as a **web application** where users can upload MRI scans and receive instant predictions.

---

## Features

* Classifies MRI scans as **Tumor** or **No Tumor**
* Built using **VGG16 Transfer Learning**
* **Image preprocessing** (resizing, normalization, augmentation) for better accuracy
*  Achieved high accuracy on test data
* **Web-based interface** for easy interaction and predictions
* Deployed and accessible online

---

##  Project Workflow

1. **Dataset Preparation**

   * MRI brain scan dataset (tumor vs. non-tumor images)
   * Preprocessing: resizing to 224×224, normalization, data augmentation

2. **Model Development**

   * Transfer learning using **VGG16** pretrained on ImageNet
   * Fine-tuning final layers for binary classification
   * Compiled with **Adam optimizer** and **binary crossentropy loss**

3. **Training & Evaluation**

   * Trained on dataset with validation split
   * Achieved high accuracy and low loss
   * Evaluated using accuracy, confusion matrix, and classification report

4. **Deployment**

   * Saved trained model as `.h5`
   * Built a **Flask/Streamlit web app**
   * User uploads MRI image → Model predicts tumor presence

---

## Tech Stack

* **Python**
* **TensorFlow / Keras** (VGG16)
* **NumPy, Pandas** (data handling)
* **Matplotlib, Seaborn** (visualization)
* **Flask / Streamlit** (deployment)

---

## Results

* Achieved **high accuracy** on test data (>90%)
* Reliable classification between tumor and non-tumor scans
* Successfully deployed as a **working web app**

---

##  How to Run Locally

1. Clone the repository

   ```bash
   git clone https://github.com/sarthak-jain03/Tumor_detection_System.git
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```


3. Run the web app

   ```bash
   streamlit run app.py
   ```

4. Upload an MRI scan and get prediction 

---

##  Project Structure

```
Brain-Tumor-Detection/
│── model/               # Trained model files (.h5)  // uploaded on my google drive since its size was greater than 100mb.
│── main.py               # Web app script (Flask/Streamlit)
│── requirements.txt     # Dependencies
│── README.md            # Project documentation
```

---

## Screenshots
<img width="1919" height="966" alt="image" src="https://github.com/user-attachments/assets/2eba7dd2-2b07-468e-b714-88c426fdd33b" />
<img width="1919" height="964" alt="image" src="https://github.com/user-attachments/assets/774d0913-3904-48d0-9c51-52d739b99cf9" />


---

## Future Improvements

* Improve dataset size & diversity
* Add explainable AI (Grad-CAM visualization of tumor regions)

---

## Acknowledgements

* MRI Brain Tumor Dataset (Kaggle/Other Source)
* VGG16 Pre-trained Weights (ImageNet)
* TensorFlow & Keras community

## Author
**Sarthak Jain**

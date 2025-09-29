# 🍿 Deep Learning for Content-Based Filtering Recommender System

## Project Description

This project implements a **Content-Based Filtering** recommender system for movies using a custom-built **Deep Neural Network (DNN)** architecture based on **TensorFlow/Keras**.

The system learns to predict a user's movie rating by generating a dense **User Feature Vector ($v_u$)** and a **Movie Feature Vector ($v_m$)** from complex, hand-engineered feature sets. The predicted rating is calculated as the **dot product** of these two vectors: $r_{p} = v_u \cdot v_m$.

The core achievement lies in the ground-up construction and training of the dual-tower neural network model using the Keras **Functional API**, ensuring maximum flexibility and transparency in the feature vector generation process.

---

## 🛠️ Key Technologies & Implementation Details

### **1. Model Architecture (Siamese Network with Dot Product)**

The model employs a dual-tower structure, where identical sub-networks process user and item inputs independently:

* **User Network (`user_NN`):** Takes **14 user features** (per-genre average ratings) as input.
* **Item Network (`item_NN`):** Takes **16 movie features** (year, average rating, 14 genre one-hot flags) as input.
* **Architecture:** Both networks are **Keras Sequential Models** with the following layers:
    * **Layer 1:** 256 units, **ReLU** activation.
    * **Layer 2:** 128 units, **ReLU** activation.
    * **Layer 3 (Output):** **32 units** (`num_outputs`), Linear activation.
* **Vector Normalization:** The 32-unit feature vectors ($v_u$ and $v_m$) are explicitly **L2-normalized** before their dot product.
* **Final Prediction Layer:** A **Dot product layer** combines the normalized $v_u$ and $v_m$ to produce the single-value predicted rating.
* **Implementation Note:** Developed the final model using the Keras **Functional API** for enhanced connectivity and flexibility.

### **2. Training & Data**

| Detail | Specification |
| :--- | :--- |
| **Dataset Source** | Reduced **MovieLens ml-latest-small** dataset (847 movies, 397 users, 25,521 ratings). |
| **Total Training Examples** | **50,884** (Ratings are duplicated to balance underrepresented genres). |
| **Feature Scaling** | **`StandardScaler`** for input features; **`MinMaxScaler`** for target ratings ($y$) to scale them between $\mathbf{-1}$ and $\mathbf{1}$. |
| **Loss Function** | **Mean Squared Error (`mse`)**. |
| **Optimizer** | **Adam** (`learning_rate = 0.01`). |
| **Model Performance** | Test loss of $\mathbf{\approx 0.0815}$, comparable to training loss, indicating good generalization. |

---

## 🎯 Prediction Capabilities & Use Cases

The trained model is demonstrated through key recommender system applications:

1.  **New User Cold-Start Prediction:**
    * Successfully generated relevant movie suggestions for a **new, unrated user** based solely on their defined genre preferences.
2.  **Existing User Rating Comparison:**
    * Validated model performance by comparing predicted ratings ($y_p$) against the actual historical ratings ($y$) for an **existing user**.
3.  **Finding Similar Items:**
    * Extracted all **32-dimensional movie feature vectors ($v_m$)** using the trained `item_NN`.
    * Computed item-to-item similarity using the **squared distance** metric, proving the model generates feature vectors that group movies with similar genres and themes.

---

## 🚀 How to Run

1.  Clone this repository.
2.  Ensure you have **NumPy**, **pandas**, **tensorflow** and **scikit-learn** installed:
```bash
pip install numpy pandas tensorflow scikit-learn tabulate
```
3. Open the file `Deep_Learning_Content_Filtering.ipynb` in Jupyter Notebook or JupyterLab.
4. Execute the cells sequentially to generate the predictions of movies for Different kinds of users.

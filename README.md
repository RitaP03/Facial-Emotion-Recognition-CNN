# Facial Emotion Recognition: CNN, Transfer Learning and Hyperparameter Optimization
 
Academic project developed for the **Computational Intelligence** course at the **Polytechnic Institute of Coimbra (ISEC)**, 2025/26, in a team of two (Ana Rita Pessoa and David Matos).
 
The goal is to classify human faces into five emotions (**anger, fear, happy, sad, surprise**) and to study how well a model trained on **synthetic** images generalizes to **real** ones. The project has three stages, each building on the previous one.
 
## Data
 
| Dataset | Use | Notes |
|---|---|---|
| [Basic Human Emotions](https://www.kaggle.com/datasets/bwandowando/basic-human-emotions) | Training and validation (~14.4k images, 80/20 split) | AI-generated faces |
| [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) | Test (5,834 images) | Real faces |
 
The datasets are not included in this repository. Download them from Kaggle and place them in `train/`, `validation/` and `test/` folders, each with one subfolder per class. `fix_split_80_20.py` creates the 80/20 train/validation split.
 
## Stages
 
### Stage 1: CNN from scratch
A convolutional network (convolution, batch normalization, max pooling, dropout, softmax) built with TensorFlow/Keras. Three experiments varying image size and number of epochs (64×64 / 20 epochs, 64×64 / 30 epochs, 32×32 / 20 epochs), evaluated with accuracy, recall, F1-score, AUC and confusion matrices.
 
### Stage 2: Swarm optimization of hyperparameters
Comparison of the **Gravitational Search Algorithm (GSA)** and **Particle Swarm Optimization (PSO)**, using the SwarmPackagePy library:
- on a benchmark function (Ackley, 2D and 3D);
- on two hyperparameters of the CNN (learning rate and dropout), with 4 agents, 6 iterations and 3 epochs per evaluation (about 30 hours of computation).
### Stage 3: Transfer learning and web app
- **MobileNetV2** pre-trained on ImageNet (frozen), with a small dense top layer.
- Hyperparameter search (hidden units, dropout, learning rate) with **Grid Search** (12 combinations), **Random Search** (10 trials) and **PSO** (5 particles).
- **Class weighting** on the base model, and a learning-curve study using 25%, 50%, 75% and 100% of the training data.
- **Gradio** web app: predict the emotion from an uploaded image or from the webcam.
## Results
 
| Model | Validation accuracy | Test accuracy (FER2013) |
|---|---|---|
| Stage 1: CNN from scratch (64×64, 20 epochs) | ~85% | 27.5% |
| Stage 3: MobileNetV2, default + class weighting | ~80% | 24.1% |
| Stage 3: Grid Search best | 79.2% | 23.6% |
| Stage 3: Random Search best | 79.7% | 23.5% |
| Stage 3: PSO best | 80.2% | 23.1% |
 
## Main finding and limitations
 
The models reach around 80% on validation data but only 23–27% on real images. This is the main result of the project: a **strong domain shift** between synthetic and real faces, which neither a pre-trained network nor hyperparameter tuning solved. Some points are worth keeping in mind when reading the numbers:
 
- With five classes, random guessing gives 20%, and always predicting the most frequent test class gives about 30%. The test results are therefore close to chance.
- Differences between the optimization methods are small (about 0.5 points on validation, about 1 point on test), and each configuration was run once, so they should not be read as a ranking.
- Class weighting was applied only to the default transfer-learning model, so its comparison with the optimized models also reflects other differences.
- No data augmentation was used, and the swarm optimization used few agents and iterations because of its computational cost.
- On the Ackley benchmark, the 3D results are negative, although the true minimum of the function is 0, so the benchmark implementation used should be reviewed.
Suggested next steps: data augmentation, fine-tuning part of MobileNetV2, and training with real images.
 
## Tech stack
 
Python · TensorFlow / Keras · scikit-learn · pandas · NumPy · Matplotlib · Seaborn · SwarmPackagePy · Gradio

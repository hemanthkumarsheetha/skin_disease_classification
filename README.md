# Skin Disease Classification


Hardware:

- All experiments were trained using Google Colab's Tesla K10 GPU.
- Tensorflow 2.x was utilized for data preprocessing and model training.

Experiments:

- The existing dataset was found to be unbalanced and have limited data.
- To avoid overfitting, the dataset was fine-tuned on pre-trained models trained on Imagenet for improved knowledge transfer and generalization.
- To address the issue of unbalanced classes, various methods were tried including data augmentation, focal loss inspired loss function, learning rate schedulers, class weighting and undersampling.
- Data Augmentation was applied to increase the size of the dataset and reduce overfitting, with techniques such as horizontal flip, rotation, height shift, width shift and zoom being used
- To handle unbalanced data, class weighting was employed.The class_weight function from the scikit-learn utils was used to compute the weights for each class, which were then fed to the model or the focal loss inspired loss function.
- Focal loss has been demonstrated to effectively address class imbalance issues in object detection. An extension of this concept was utilized in the form of focal sparse categorical crossentropy, which was inspired by focal loss and used as the loss function. I used focal-loss open-source library to add to loss function for training.
- To stabilize training and address unbalanced data, the use of learning rate schedulers is recommended. For some experiments, an exponential learning rate decay was employed.
- Undersampling is another solution for handling class imbalance. In this approach, the number of images for each class was matched to the lowest class count(16 in our case).

- For fine-tuning, I tried the following pre-trained models: MobileNetV3,ResNet50 and EfficientNet.
- Early Stopping was used to tackle varying validation losses.

Evaluation Metrics:

- For evaluation, recall,f1 score, precision and confusion matrix were used. Recall was given higher preference.
- AUC was also used for some experiments

Results:

- Among all of the variation of the methods described above, Data Augmentation + Fine-tuned MobileNetV3 + focal_sparse_ce_loss(with Adam optimization) trained with the given unbalanced data(112 images in 3 classes) performed the best in terms of performance and metrics.
-  Other variations didn't yield as compelling results. Some of them were either overfitting or performed worse or were unstable.

Training and Evaluation Code:

- Please take a look at my oro_health_training.ipynb file for reviewing my training experiments
- Please run evaluation by running my evaluation.ipynb which makes use mobilenet.zip file

Classification Report:
<br />
On GPU: 
<br />
![Classification_report_gpu](gpu_classification_report.png)
![confusion_matrix](confusion_matrix.png)

On CPU:
![Classification_report_gpu](cpu_classification_report.png)
![confusion_matrix](cpu_confusion_matrix.png)

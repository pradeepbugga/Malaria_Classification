# Malaria Cell Image Classification with CNNs and Saliency Maps
A convolutional neural network (CNN) for classifying parasitized ("infected") vs "uninfected" thin blood smears
Includes K-fold cross validation, MLflow experiment tracking, and explainability via contrastive saliency maps 
<p align="center">
<img width="750" height="300" alt="image" src="https://github.com/user-attachments/assets/1013ef33-12cd-4e8c-ab17-20d0b63e8e55" />
</p>  


## *Problem Statement*

Current malaria detection from thick blood smears is a manual, time-consuming process requiring medical expertise in low-resource settings.
Deep learning approaches can standardize classification while also providing cost and time savings.

## *Project Summary*
Our final model ("cnn_rgbs_5l") is a 5-layer convolutional neural network using input RGB images feature-engineered with the S channel of HSV color space (i.e. RGB+S).
While our model demonstrated great performance, it is limited by label noise (incorrectly annotated images or images with artifacts).
The final deliverable of this project is an html gallery(https://pradeepbugga.github.io/Malaria_Classification/ ) showing these edge cases and the model's acknowledgement of any potential parasitic signatures.  An annotator can use this gallery to re-label edge cases, enabling a fully clean dataset for an ultra-accurate ML model.


## *Dataset Description*
The thin blood smear images can be found at the National Library of Medicine (NLM) (https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html)
There are 27,558 total annotated RGB images (half "infected", half "uninfected")
For model development, 2,600 images are held out as a test set (1300 each class).  

The datasets are placed in ./data/cell_images/train or ./data/cell_images/test,  and within each of those folders is a sub-folder called "parasitized" or "uninfected" <br><br>

## *Data Pre-Processing/Loading*
./src/preprocessing.py contains functions for converting the RGB images to either HSV or CLAHE image arrays
./src/visualize.py contains functions for visualizing the RGB images converted to HSV, CLAHE, or RGBS (RGB + S channel of HSV) (to be used with ./scripts/visualize_*.py) 

./src/data_loader.py contains the functions for loading the dataset via data generator for model training
Within these functions is also augmentation and feature engineering (RGB-> RGBH or RGB-> RGBS)

## *Model Architecture*
./src/model_builder.py contains all the model architectures used in development   <br><br>
./scripts/train_cnn*.py trains the corresponding models, then outputs results in the ./models folder (model.keras, /logs/history, /reports/classification report, /reports/confusion matrix, /predictions/predictions CSV) <br><br>
./scripts/train_cnn_rgbs_5l.py corresponds to our ultimate model that achieved 99% accuracy, precision, and recall <br><br>

## *Training Analysis*
./src/visualize.py contains functions for plotting accuracy/loss from history and the confusion matrix (to be run with ./scripts/analyze_history.py and /visualize_cm.py) <br><br>
./src/analyze.py contains functions for plotting the extended confusion matrix (separated by 0.1 wide probability bins) (to be run with ./scripts/extended_cm.py) -> this outputs to ./models/reports 

## *False Positive/Negative Analysis*
./src/analyze.py also contains functions for analyzing false positives and negatives <br><br>
./scripts/false_neg_pos_logger.py logs all false positives and negatives to a .json file in ./models/misclassification <br><br>
./scripts/false_neg_pos_viewer.py allows visualization of false positive / false negative / true negative / true positive images by the actual label and predicted probability by bin (i.e. false positive in label = 0, predicted prob = 0.9-1.0) <br><br>

## *AI Explainability Analysis*
./src/gradcam_utils.py has functions for either Grad-CAM, saliency, or contrastive saliency. <br><br>
./scripts/run_gradcam.py or /run_saliency.py or run_contrastive_saliency.py generate the corresponding maps given an image path (these output to ./models/explainability) 

## *Out-of-Fold Predictions (Label Noise Analysis)*
To identify all edge cases / potential mis-labels from the dataset, we perform k-fold validation, holding out a different fraction of the dataset for out-of-fold predictions. We can group all the results at the end to get an understanding of the entire dataset <br><br>
./scripts/kfold_train_rgbs_5l.py performs this process, merging the train and test folders, using the optimized cnn_rgbs_5l model from ./src/model_builder.py and a new data generator at ./src/data_loader.py <br><br>
The results are then outputted to folders corresponding to each fold (i.e. ./models/kfold_cnn_rgbs_5l/fold_*) <br>

Next, we run ./scripts/false_neg_pos_logger.py to generate .json logs of all FP's and FN's for all 5 folds. (in ./models/kfold_cnn_rgbs_5l/fold_*/misclassification/)
./scripts/false_pos_neg_df.py then converts those .json logs into a Pandas dataframe, concatenates, then outputs to a .csv in ./models/kfold_cnn_rgbs_5l

Next, we run ./scripts/false_pos_neg_saliency.py to generate contrastive saliency maps on each FP/FN image in the Pandas dataframe, saving the images in ./models/kfold_cnn_rgbs_5l/fold_*/explainability/saliency_maps, then adding a column in the df for the image paths

Finally, we run ./scripts/html_gen.py to generate a gallery of false positives and negatives as an HTML file.  This gallery has the predicted probability, rank (by probability), and the side-by-side of the original image and contrastive saliency map
These two htmls (one for false positive and one for false negative) are displayed on GitHub Pages and can be accessed at https://pradeepbugga.github.io/Malaria_Classification/ 

## *MLflow Logging*
We retroactively log our models using ./scripts/log_existing_models.py to MLflow, specifically noting # false positives, negatives, f1-score, accuracy, precision, and recall.  This script logs models to a local MLflow server. 














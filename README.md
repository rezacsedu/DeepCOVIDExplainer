## DeepKneeExplainer
Supplementary materials for "COVIDExplainer: Explainable COVID-19 Diagnosis from Chest Radiography Images" submitted to ECML-PKDD'2020. We provide details of dataset, preprocessing, network architectures, and some additional results. Nevertheless, we'll provide trained models, preprocessed data, interactive Python notebooks, and a web application showing live demo. As planned, we keep this repo updated. 

### Methods
The pipeline of "COVIDExplainer" consist of preprocessing, classification, snapshot neural ensemble, and decision visualizations.
After necessary preprocessing of CXR images, DenseNet, ResNets, and VGGNets are trained in a transfer learning setting, creating their model snapshots, followed by neural snapshot ensemble based on averaging Softmax class posterior and the prediction maximization of best performing models. Finally, class-discriminating attention maps are generated using gradient-guided class activation maps (Grad-CAM++) and layer-wise relevance propagation (LRP) to provide explanations of the predictions and to identify the critical regions on patients chest.  

### Datasets
We choose 3 different versions of COVIDx dataset to train and evaluate the model. The COVIDx v1.0 had a total of 5,941 CXR images from 2,839 patients. It is based on COVID-19 image dataset curated by Joseph P. C., et al. and Kaggle CXR Pneumonia dataset (https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) by Paul Mooney. COVIDx v1.0 is already used in a literature (https://arxiv.org/abs/2003.09871). However, Kaggle CXR images are of children. Therefore, to avoid possible prediction bias~(e.g., the model might predict based on the chest size itself), we consider using the CXR of adults with pneumonia by augmenting more CXR images from COVID-19 confirmed cases.

COVIDx v2.0 dataset is also based on COVID-19 image dataset, but come with RSNA Pneumonia Detection Challenge dataset (https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) provided by the Radiological Society of North America. On the other hand, COVIDx v3.0 is also based on COVID-19 image dataset and RSNA Pneumonia dataset, we enriched it with additional 49 COVID-19 CXR from: i) Italian Radiological Case CASE (https://radiopaedia.org/articles/covid-19-3?lang=us), and ii) Radiopaedia.org (provided by Dr. Fabio Macori)(https://www.sirm.org/category/senza-categoria/covid-19/). COVIDx v1.0 CXR images are categorized with normal, bacterial, non-COVID19 viral and COVID19 viral, whereas COVIDx v2.0 and v3.0 CXR images are categorized as normal, pneumonia and COVID19 viral. The distribution of images and patient cases amongst the different infection types are as follows: 


### Data availability


### Availability of pretrained models
We plan to make public all the pretrained models and some computational resources available, but it will take time. For the time being, we made only VGG-19 and ResNet-18 upon reasonable request. 

### A quick instructions
A quick example on a small dataset can be performed as follows: 
* $ cd GradCAM_FI
* $ python3 load_data.py (make sure that the data in CSV format in the 'data' folder)
* $ python3 model.py
* $ python3 grad_cam.py

### Citation request
If you use the code of this repository in your research, please consider citing the folowing papers:

    @inproceedings{COVIDExplainer,
        title={COVIDExplainer: Explainable COVID-19 Diagnosis from Chest Radiography Images},
        author={Anonymized for review},
        conference={ECML-PKDD'2020},
        publisher={Under review},
        year={2020}
    }

### Contributing
In future, we'll provide an email address, in case readers have any questions. 

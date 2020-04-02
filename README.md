### DeepKneeExplainer
Supplementary materials for "COVIDExplainer: Explainable COVID-19 Diagnosis from Chest Radiography Images" submitted to ECML-PKDD'2020. We provide details of dataset, preprocessing, network architectures, and some additional results.

#### Methods

#### Data collections


#### Data availability


#### Availability of pretrained models
We plan to make public all the pretrained models and some computational resources available, but it will take time. For the time being, we can make VGG19 and ResNet18 upon request. 

#### A quick instructions
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

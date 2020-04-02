### DeepKneeExplainer
Code and supplementary materials for our paper titled "DeepKneeExplainer: Explainable Knee Osteoarthritis DiagnosisBased on Radiographs and Magnetic Resonance Images", submitted to Elsevier Artificial Intelligence journal. There are only codes with simple file structures. Other codes, trained models and processed images are in the path /data/jiao of our GPU server. There would be another ReadMe file. So codes uploaded here are separated by steps in the pipeline. In each Python file, there are several comments so that you can know where to change.

#### Methods
In general, first run codes for different modalities in Preprocessing folder, then you get processed images, which are the inputs of ROI step. ROI steps also need labelled bounding boxes. After ROI steps, you'll get the segmentation results, which can be dealt with function cv2.findContours() and boundingRect() in OpenCV. Alternatively, it can be done with FRCNN. From extracted ROIs, you can run classifier codes, then you will get trained models, which can be run for Grad-CAM, Grad-CAM++, and LRP visualizations. For classifiers, run VGG_new.py for VGG, ResNet_newJSN.py for ResNet and DenseNet_newJSN.py for DenseNet.

#### Data collections
Radiographs, 2D MRI slices, and ground truths covering 3,026 subjects and their six follow-up examinations are collected (link: http://most.ucsf.edu/) from the MOST study. Out of 3,026 knee radiographic assessments, only 2,406 mutual patients were engaged in the MRI collection. To mitigate possible decision bias and out-of-distributions, our study did not consider other dataset like The Osteoarthritis Initiative (OAI) (link: https://oai.epi-ucsf.org/datarelease/). However, to maintain data integrity, this study and the subsequent evaluation are based on the first visit data which provide MRI slices from 4 perspectives: 4,812 radiographs from the coronal plane, 4,748 radiographs from the sagittal plane, 4,676 MRI slices from the axial plane, and 4,678 MRI slices.

MOST provides 3 types of semi-quantitative scoring systems: KL scale, OARSI JSN progression gauged from the medial tibiofemoral compartment, and OARSI JSN progression assessed from the lateral tibiofemoral compartment. Considering the prevalence of KL in automatic MRI quantification and multimodality integration, we assign the same radiographic semi-quantitative labels for both radiographs and MRIs. However, JSN progressions of lateral tibiofemoral compartments are excessively imbalanced, and hence only the first two grading schemes were adopted.

#### Data availability
Due to NDA, we're not allowed to share the original data. Please contact Multicenter Osteoarthritis Study (MOST). Link: http://most.ucsf.edu/}

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

    @inproceedings{karim2020DeepKneeXAI,
        title={DeepKneeExplainer: Explainable Knee Osteoarthritis DiagnosisBased on Radiographs and Magnetic Resonance Images},
        author={Karima,Md. Rezaul and Jiao, Jiaob and Döhmena, Till and Rebholz-Schuhmannd, Dietrich and Deckera, Stefan and Cochez, Michael and Beyan, Oya},
        journal={Artificial Intelligence},
        publisher={Elsevier (under review)},
        year={2020}
    }

### Contributing
For any questions, feel free to open an issue or contact at rezaul.karim@rwth-aachen.de

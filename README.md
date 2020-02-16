# GestRecog
---
This is used for the project "ZenS" of organization CRI.

Initially, for input we have a dataset of 288 samples (8 types * 36 samples). Each sample has 256 sequence of 6-axis data.

### Workflow for dataset construction 
1. Original dataset collection.
2. interception: Truncate the data sequence to 6 * 256.
3. preprocessing: noise reduction, zero mean, normalization, PCA/Whitening.
4. done: new dataset.

### Workflow for training
1. import dataset.
2. train the neuro network.
3. save model.

### Workflow for application
1. getting original data sequence and intercept it simultaneously.
2. preprocessing: noise reduction, zero mean, normalization, PCA/Whitening.
3. forward propagation.
4. done: return a string representing the type of gesture.
5. inter-program communication: send the result to Unity3D program.

### Structure
- /application: added to ZenS as a tool for gesture recognition.
    - data_gotten: stores the raw data of gestures obtained from the launcher, 
    in addition to the files being recognized or about to be identified, 
    there are also historical records (raw data).
    - results:  there are historical results of recognition.
- /dataset: for training only. For test, it will be in the /testing.
    - new: pre-processed dataset.
    - original: raw dataset.
- /dataset_collection: organize raw dataset and convert it to pre-processed dataset.
- /models: trained models.
- /preprocessing: preprocess the data.
- /reference: papers
- /testing: test the models. There will be a dataset for testing only in it.
- /training: train the models.
- /.gitignore
- /README.md
    

### Methodology
1. the methodology above is based on  **numerical** data
2. we are accouting for another methodology who is based on **image** data.
In the methodology, we convert the sequence of numerical data to its image connecting the discreted data. 
And the problem of Gesture Recognition become an image recognition problem to a certain extent.
The second Methodology is in the repository [GestRecog_usingImage](https://github.com/ZenMoore/GestRecog_usingImage)
# GestRecog
---
This is used for the project "ZenS" of organization CRI.

Initially, for input we have a dataset of 288 samples (8 types * 36 samples). Each sample has 256 sequence of 6-axis data.

### Workflow of dataset construction 
1. Original dataset collection.
2. interception: Truncate the data sequence to 6 * 256.
3. preprocessing: noise reduction, zero mean, normalization, PCA/Whitening.
4. done: new dataset.

### Workflow of training
1. import dataset.
2. train the neuro network.
3. save model.

### Workflow of application
1. getting original data sequence.
2. interception: Truncate the data sequence to 6 * 256.
3. preprocessing: noise reduction, zero mean, normalization, PCA/Whitening.
4. forward propagation.
5. done: return a string representing the type of gesture.
6. inter-program communication: send the result to Unity3D program.

### Methodology
1. the methodology above is based on  **numerical** data
2. we are accouting for another methodology who is based on **image** data.
In the methodology, we convert the sequence of numerical data to its image connecting the discreted data. 
And the problem of Gesture Recognition become an image recognition problem to a certain extent.
The second Methodology is in the repository GestRecog_usingImage
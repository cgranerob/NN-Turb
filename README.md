# NN-Turb
Neural network stochastic model generating 1-dimensional fields with turbulent velocity statistics

We define and study a fully-convolutional neural network stochastic model, NN-Turb, which generates 1-dimensional fields with turbulent velocity statistics. Thus, the generated process satisfies the Kolmogorov 2/3 law for second order structure function. It also presents negative skewness across scales (i.e. Kolmogorov 4/5 law) and exhibits intermittency. Furthermore, our model is never in contact with turbulent data and only needs the desired statistical behavior of the structure functions across scales for training.

# Contents of Thunmpy package
This repository contains :

1) The Neural network stochastic NN-Turb model: NN-Turb.pt file.
2) A file with the model definition and the training setup: NN-Turb_Training.py
3) A notebook with some code to load the model, generate some realizations of the 1D stochastic field with turbulent velocity statistics and plot some figures: NN-Turb_Fields_Generation.ipynb
4) A file with the functions needed to compute the used loss during training : analyseIncrsTorchcuda.py
5) A file with the reference behaviors for the second order structure function, skewness and flatness: Train_TurbModane.npz

# References
Granero-Belinchon, C. **Neural network based generation of 1-dimensional stochastic fields with turbulent velocity statistics**. Hal/ArXiv. 2022. <a href="[link](https://hal.archives-ouvertes.fr/hal-03861273)" > [link](https://hal.archives-ouvertes.fr/hal-03861273) </a>.

# Contact

Carlos Granero-Belinchon <br />
Mathematical and Electrical Engineering Department <br />
IMT Atlantique <br />
Brest, France <br />
e: carlos.granero-belinchon [at] imt-atlantique.fr <br />
w: https://cgranerob.github.io/ <br />
w: https://www.imt-atlantique.fr/en/person/carlos-granero-belinchon <br />



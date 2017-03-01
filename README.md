# DAAE

Denosing Adversairal Autoencoder

This repo contains the code needed to run the experiments from the paper "Denoising Adversarial Autoencoders", Antonia Creswell and Anil Anthony Bharath

We provide examples of experimental results for each of our networks (DAAE and iDAAE) trained on 3 datasets. We do not (yet) provide the trained models but do provide the network and training parameters needed to replicate our results. Note that results, expecially for the log-likelihood may vary because of the stoachstic nature of the image generation process.

#To use the code:
1. clone the repo
2. unzip all the compressed data files
3. install lasagne, theano, matplotlib, scikit-image, numpy, dill, ipython, jupyter, scikit-learn ... (a full list may be found it reqiurements.txt)
4. run the following from comand line to start a notebook: 
	$ ipython notebook
5. run the code by pressing the "play" (triangle |>) button

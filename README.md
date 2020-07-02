# CCACSP

This is an implementation of the canonical correlation approach to common spatial patterns (CCACSP) algorithm proposed by Eunho Noh and Virginia de Sa. "Canonical correlation approach to common spatial patterns." 2013 6th International IEEE/EMBS Conference on Neural Engineering (NER). IEEE, 2013.

This code provides a set of filters with the goal of improving discriminability in motor imagery brain-computer interfaces (MI-BCI). Aside from learning the CCACSP filters, this code also provides functions to perform classification. 

## How to use

You should have numpy, scipy and scikit-learn installed. List of avaibale functions include `calc_CCACSP`, `train` and `test`. You can either use `calc_CCACSP` by itself or use it via the `train` and `test` functions to perform feature extraction, classification and test. If you would like to use the `calc_CCACSP` only, here is an example. Note that `numFilt` specifies the number of filters for each class:

```python
import CCACSP   # CCACSP.py should be in your directory 

# train_data_0 and train_data_1 contain the training data belonging to classes 0 and 1 each with the following format: 
# [number of trials, number of channels, time samples]
# outputs filters of size [number of channels, 2xnumFilt]
filers = CCACSP.calc_CCACSP(train_data_0, train_data_1, numFilt)

```

The following explains how to use the `train` and `test` functions. Note that `numFilt` specifies the number of filters for each class. The classifier defined in line 21 of the code is an LDA with auto shrinkage. You can replace it with your choice of classifier from scikit-learn or elsewhere. 

```python
import CCACSP   # CCACSP.py should be in your directory 

# train_data_0 and train_data_1 contain the training data belonging to classes 0 and 1 each with the following format: 
# [number of trials, number of channels, time samples]
# outputs filters and the classifier  
filers, clf = CCACSP.train(train_data_0, train_data_1, numFilt)

# test_data is a single trial with the following format: [number of channels, time samples]
# output = 0 or 1 for belonging to classes 0 and 1 respectively 
output = CCACSP.test(test_data, filers, clf)
```

### Citation

If you use this code, please cite the following paper:

Eunho Noh and Virginia de Sa. "Canonical correlation approach to common spatial patterns." 2013 6th International IEEE/EMBS Conference on Neural Engineering (NER). IEEE, 2013.
http://www.cogsci.ucsd.edu/~desa/CCACSP.pdf

### Questions/Comments 
Please send any questions or comments to mahta@ucsd.edu or mahta.mousavi@gmail.com

Thank you! 

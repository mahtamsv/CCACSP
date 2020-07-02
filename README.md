# CCACSP

This is an implementation of the canonical correlation approach to common spatial patterns (CCACSP) algorithm proposed by Eunho Noh and Virginia de Sa. "Canonical correlation approach to common spatial patterns." 2013 6th International IEEE/EMBS Conference on Neural Engineering (NER). IEEE, 2013.

Aside from learning the CCACSP filters, this code also provides functions to perform classification. 

Please note that the code does not perform any extra steps to accomodate for unbalanced train or test input datasets. If you have unbalanced datasets, the code will not encounter any errors but the interpretation of the results will be difficult. Therefore, it is recommended that you simply balance the classes by randomly subsampling the larger class. To minimize loosing data, you can perform balancing multiple times and run the code and then take the average. 

## How to use

You should have numpy, scipy and scikit-learn installed. 

List of avaibale functions `calc_CCACSP`, `train` and `test`. 

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

### Questions/Comments 
Please send any questions or comments to mahta@ucsd.edu or mahta.mousavi@gmail.com

Thank you! 

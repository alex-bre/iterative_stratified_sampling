# Stratified Sampling
This iterative stratified sampling method matches the distribution of discrete/categorical variables from a given 
dataset to a specified reference dataset by iteratively removing samples. The minimal percentage of data that should be 
contained in the new dataset can be passed via an argument, the default is set to 80%. If exact matching is specified 
via the arguments, the method will attempt to exactly match the distribution and remove as many samples as needed.
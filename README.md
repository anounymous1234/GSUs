# Gated Spectral Units
* **Tensorflow 1.12.0**
* Python 3.6.9
* CUDA 9.0+ (For GPU)

# Instructions
1. download amazon data from http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Digital_Music_5.json.gz
and
http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video_5.json.gz

2. Extract and place them into the corresponding folder.

3. To generate data, go to one of folders and run preprcess file. For example:
```
$ cd data/movielens_1m
$ python preprocess.py

4. To run the experiment with default parameters:
```
$ cd codes
$ python main.py
```

This repo serves to repeat all experimental results. You can change all the parameters in `codes/params.py`.




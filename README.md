# Relation Extraction - RDD Corpus

This document explains the details for the reproduction of the results obtained with the RDD corpus in the task of detecting relationships between rare diseases and disabilities[2].

## Directory Structure
This repository is divided as follows:

	.
	├── src 
	│	|── data	
	|	|	|── files
	│	|	|	└── train-todo.txt (Text file with the corpus)
	|	|   	|──  pkl (Intermediate directory with the preprocessed information.)
	│	│	|	└── *.pk 
	|	│	└── configurations
	│	|	│	└── configuration_*.json
	│	├── RDD-CNN.py (File which contains the routines for creating and training the model)
	│	├── preprocess.py (Script for corpus preprocessing)    
	├── trained-model (or build)
	|	├── RDD-CNN.h5
	|	├── predictions.txt
	|	└── real.txt
	├── Readme.md
	└── requirements.txt

## Corpus RDD
To carry out this experiment we have used the relationships file provided in the RDD corpus. This file includes annotations about relationships between disabilities and rare diseases that appear in the different sentences. Each line follows the following format.

1. The first column is the label, which may be:
	1. rd-dis: if the sentence expresses a relationship between disability and rare disease
	2. none: if the sentence does not clearly and unambiguously express a relationship between disability and rare disease.
2. The second column indicates the position where the rare disease begins. 
3. The third column indicates the position where the disability begins.
4. The last column is the tokenized sentence.


An example can be found below
```
rd-disab 0 11 Leber congenital amaurosis LCA is one of the main causes of childhood blindness .
rd-disab 0 12 Fragile X syndrome FXS is the most frequent inherited form of human mental retardation .
```
This file is located at: src/data/files/train-todo.txt


## Instructions for running the experiment
For this experiment Dependency-Based Word Embeddings has been used [1]
```bash
curl http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2 --output deps.words.bz2
bzip2 -d deps.words.bz2
rm -rf data/levy_word_emb && mkdir data/levy_word_emb
mv deps.words data/levy_word_emb
```
To run the experiment
```bash
python preprocess.py
python RDD-CNN.py
```
The model shown here has the following configuration:
```python
experiment_configuration = {
 		"batch_size" 	: 128,	
		"nb_filter" 	: 100,
		"filter_length"	: 3,
		"hidden_dims" 	: 100,
		"nb_epoch" 	: 20,
		"position_dims"	: 50,
		"seed":1578
}
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
merge_10 (Merge)             (None, 88, 500)           0         
_________________________________________________________________
conv1d_19 (Conv1D)           (None, 88, 100)           150100    
_________________________________________________________________
global_max_pooling1d_19 (Glo (None, 100)               0         
_________________________________________________________________
reshape_10 (Reshape)         (None, 10, 10)            0         
_________________________________________________________________
conv1d_20 (Conv1D)           (None, 8, 100)            3100      
_________________________________________________________________
global_max_pooling1d_20 (Glo (None, 100)               0         
_________________________________________________________________
dropout_10 (Dropout)         (None, 100)               0         
_________________________________________________________________
dense_10 (Dense)             (None, 2)                 202       
=================================================================
Total params: 160,502
Trainable params: 159,702
Non-trainable params: 800
_________________________________________________________________

```

### Making predictions
Set of instructions for classifying a sentence:
```python
from keras.preprocessing import sequence
from keras.models import load_model
import gzip
import cPickle as pkl
import numpy as np

# Instructions for loading the pre-trained model
model = load_model('trained-model/RDD-CNN.h5')

# Load dictionaries
word2Idx = pkl.load(gzip.open('src/data/pkl/utils.pkl.gz', 'rb'))

sentence = "Autism in CHARGE association may represent a neuro endocrine dysfunction ."
pos_dis = 0 # AUSTISM
pos_rd = 2 # CHARGE
# Translation of terms based on dictionaries - 88 is the maximum sentence length allowed by the experiment
tokens = sentence.split(" ")
positionValues1 = [posToValue(idx,pos_dis,distanceMapping,-30) for idx in xrange(0, len(tokens))]
positionValues2 = [posToValue(idx,pos_rd,distanceMapping,-30) for idx in xrange(0, len(tokens))]
trad_tokens = [word2Idx[x] if x in word2Idx else word2Idx["UNKNOWN"] for x in tokens]


trad_tokens = sequence.pad_sequences([trad_tokens],maxlen=88,padding='post',value=word2Idx['PADDING'])

positionValues1 = sequence.pad_sequences([positionValues1],maxlen=88,padding='post',value=distanceMapping['PADDING'])
positionValues2 = sequence.pad_sequences([positionValues2],maxlen=88,padding='post',value=distanceMapping['PADDING'])

rel = np.argmax(model.predict([trad_tokens,positionValues1,positionValues2],verbose=0),axis=1)[0]

print(sentence,"rd-disab" if rel==1 else "None")

```

```
Out[57]: 
('Autism in CHARGE association may represent a neuro endocrine dysfunction .', 'rd-disab')
```

## Results

We have evaluated our model using a 10-fold cross validation.

|                 	|           	| Evaluation 1 	|           	|
|-----------------	|-----------	|--------------	|-----------	|
|                 	| Precision 	| Recall       	| F-measure 	|
| CNN-W+POS       	| 75.76     	| 75.90        	| 75.57     	|
-----------------------------------------------------------------

## References

[1] -  Levy, O., & Goldberg, Y. (2014). Neural word embedding as implicit matrix factorization. In Advances in neural information processing systems (pp. 2177-2185).

[2] -  Lourdes Araujo, Hermenegildo Fabregat Marcos, Juan Martinez-romo (2018). Deep neural models for extracting entities and relationships in the new RDD corpus relating disabilities and rare diseases (In revision)




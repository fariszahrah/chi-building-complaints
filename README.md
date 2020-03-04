#chi-building-complaints

# How does the City of Chicago prioritize building complaint responses?

What predictable information does the complaint itself have?

> Language - the actual content of the complaint

> Location - Neighborhood, etc.

> Complaint type

> Street type


### Install

> pip3 install -r src/requirements.txt

## Models Tested 

> Decision Tree Classifier (sklearn)

> Random Forest

> CNN  -- Only using the specific complaint detail sentences, Pre-trained word vectors downloaded from [GloVe](https://nlp.stanford.edu/projects/glove/).  100-dimension vectors were used




### Some interesting things to note


> When we are not using the Convnet on the actual content of the complaint we are able to analyze the feature importances much easier.  A good thing to note for chicago is that the primary feature that indicates whether the city will enforce a complaint is whether or not the building has a permit(construction,demo,etc).  The CNN classifier outperforms the decsion trees, with a dropout of p=0.5

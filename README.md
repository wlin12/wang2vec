# wang2vec
Extension of the original word2vec (https://code.google.com/p/word2vec/) using different architectures

To build the code, simply run:

make

The command to build word embeddings is exactly the same as in the original version, except that we removed the argument -cbow and replaced it with the argument -type:

./word2vec -train <input_file> -output <output_file> -type 0 -size 50 -window 5 -negative 10 -hs 0 -sample 1e-4 -threads 1 -binary 1 -iter 5

The -type argument is a integer that defines the architecture to use. These are the possible parameters:  
0 - cbow  
1 - skipngram  
2 - cwindow (see below)  
3 - structured skipngram(see below)  
4 - collobert's senna context window model (still experimental)  

If you use functionalities we added to the original code for research, please support us by citing our paper (thanks!):

@InProceedings{Ling:2015:naacl,  
author = {Ling, Wang and Dyer, Chris and Black, Alan and Trancoso, Isabel},  
title="Two/Too Simple Adaptations of word2vec for Syntax Problems",  
booktitle="Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",  
year="2015",  
publisher="Association for Computational Linguistics",  
location="Denver, Colorado",  
}

The main changes we made to the code are:

****** Structured Skipngram and CWINDOW ******

The two NN architectures cwindow and structured skipngram (aimed for solving syntax problems). 

These are described in our paper:

-Two/Too Simple Adaptations of word2vec for Syntax Problems

****** Class-based Negative Sampling ******

A new argument -negative-classes can be added to specify groups of classes. It receives a file in the format:
 
N dog  
N cat  
N worm  
V doing  
V finding  
V dodging  
A charming  
A satirical  

where each line defines a class and a word belonging to that class. For words belonging to the class, negative sampling is only performed on words on that class. For instance, if the desired output is dog, we would only sample from cat and worm. For words not in the list, sampling is performed over all word types.

warning: the file must be order so that all words in the same class are grouped, so the following would not work correctly.

N dog  
A charming  
N cat  
N worm  
V doing  
V finding  
V dodging  
A satirical  

****** Minor Changes ******

The distance_txt and kmeans_txt are adaptations of the original distance and kmeans code to take textual (-binary 0) embeddings as input

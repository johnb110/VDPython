# VDPython
VulDeePecker algorithm implemented in Python  

## VulDeePecker
* Detects exploitable code in C/C++ 
* Uses N-grams and deep learning with LSTMs to train detection model
* Invents idea of code gadgets for semantically-related code
  * Code gadgets are vectorized for input to neural network
  * [Training/testing set for this project includes existing code gadgets and vulnerability classification]
* Trained on two vulnerability types
* [Paper](https://arxiv.org/pdf/1801.01681)
* [GitHub](https://github.com/CGCL-codes/VulDeePecker)

## Running project
* To run program, use this command: `python vuldeepecker.py [gadget_file]`, where gadget_file is one of the text files containing a gadget set
* Program has 3 parts:
  * Performing gadget "cleaning"
    * Remove comments, string/character literals
    * Replacing all user-defined variables and functions with VAR# and FUN#, respectively
      * The # is an integer identifying the user-defined variable/function within the gadget
      * Note: this identifier only applies within the scope of the gadget
  * Vectorize gadget
    * Gadgets are parsed, tokenized, and transformed to vectors of embeddings
    * Vectors are normalized to a constant length through either truncation or padding
  * Train and test neural model
    * Gadget vectors are used as input to train the neural model 
    * Data is split into training set and testing set
    * Neural model is trained, tested, and accuracy is reported

## Code Files
* vuldeepecker.py
  * Interface to project, uses functionality from other code files
  * Fetches each gadget, cleans, buffers, trains Word2Vec model, vectorizes, passes to neural net
* clean_gadget.py
  * For each gadget, replaces all user variables with "VAR#" and user functions with "FUN#"
  * Removes content from string and character literals
* vectorize_gadget.py
  * Converts gadgets into vectors
  * Tokenizes gadget (converts to symbols, operators, keywords)
  * Uses Word2Vec to convert tokens to embeddings
  * Combines token embeddings in a gadget to create 2D gadget vector
* blstm.py
  * Defines Bidirectional Long Short Term Memory neural network for training/prediction of vulnerabilities
  * Gets gadget vectors as input
  * Implements functions for both training and testing the model
  * Uses parameters defined in VulDeePecker paper

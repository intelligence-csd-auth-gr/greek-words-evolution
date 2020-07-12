# Greek Words Evolution

This repository accompanies the paper "Studying the Evolution of Greek Words via Word Embeddings" by V. Barzokas, E. Papagiannopoulou and G. Tsoumakas, published in the proceedings of the 11th Hellenic Conference on Artificial Intelligence (SETN 2020) and contains the set of tools developed and data prepared for its needs. If you use this code and/or data in your research plase cite the following: V. Barzokas, E. Papagiannopoulou and G. Tsoumakas (2020) "Studying the Evolution of Greek Words via Word Embeddings", In Proceedings of the 11th Hellenic Conference on Artificial Intelligence (SETN 2020).


## Requirements
* [Python 3.6.9](https://www.python.org/downloads/release/python-369/)
* [fastText](https://fasttext.cc/) - a library for efficient learning of word representations and sentence classification.

## Installation
0. Clone this repository by running:

    ```
    git clone git@github.com:intelligence-csd-auth-gr/greek-words-evolution.git
    ```
   
0. Clone the required `fastText` repository by running:

    ```
    git submodule init
    git submodule update
    ```
   
0. Install the `fastText` library for your system as described in its documentation that can be found here: https://github.com/facebookresearch/fastText

    **Note:** Normally all that is required to do is:
    
        cd fastText
        make
        pip install .
            
0. Install the required Python libraries by running:

    ```
    pip install -r requirements.txt
    ```

## Running
### First steps
If running for first time, create the text files per period by running:
    
    python index.py metadata --exportTextByPeriod --corpusName openbook
    python index.py model --action create

Later, after the models have been generated you can see the nearest neighbours of a word by running something similar to this example:
    
    python index.py model --action getNN --word αισθάνομαι --period 1800

Get the 10 words with the highest semantic change, based on their cosine distance:
        
    python index.py model --action getCD --fromPeriod 1800 --toPeriod 1900
                
Get the 10 words with the highest semantic change, based on their cosine similarity (opposite sorted list of cosine distance):
        
    python index.py model --action getCS --fromPeriod 1800 --toPeriod 1900

### Options 
In order to see a full list of the available options and a short description of each one of them, type:

    python index.py --help

## License

[Apache License 2.0](LICENSE)

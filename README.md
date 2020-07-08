# Greek Words Evolution

### Requirements
* Python 3.6.9
* fastText - a library for efficient learning of word representations and sentence classification.

### Installation
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

### Running
If running for first time, create the text files per period by running:
    
    python index.py metadata --exportTextByPeriod --corpusName openbook
    python index.py model --action create

Later, after the models have been generated you can see the nearest neighbours of a word by running something similar to this example:
    
    python index.py model --action getNN --word αισθάνομαι --period 1800

Get the 10 words with the highest semantic change, based on their cosine distance:
        
    python index.py model --action getCD --fromPeriod 1800 --toPeriod 1900
                
Get the 10 words with the highest semantic change, based on their cosine similarity (opposite sorted list of cosine distance):
        
    python index.py model --action getCS --fromPeriod 1800 --toPeriod 1900

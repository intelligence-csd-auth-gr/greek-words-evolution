# Greek Words Evolution

### Requirements
* Python 3.6.9
* fastText - a library for efficient learning of word representations and sentence classification.

### Installation
0. Clone this repository by running:

    ```
        git clone git@github.com:vbarzokas/greek_words_evolution.git
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
If running for first time, create the text files per decade by running:
    
        python index.py metadata --exportTextByDecade
        python index.py model --action create

Later, after the models have been generated try to see the nearest neighbours of a word by running something similar to this example:
    
        python index.py model --action getNN --word αισθάνομαι --decade 1800
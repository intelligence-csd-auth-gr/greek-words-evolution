# Greek Words Evolution

## Overview
A systematic framework that uses diachronic word embeddings to trace semantic shifts or variations in the context of words over time in the Greek language.

##### Research paper
This repository accompanies the paper _"Studying the Evolution of Greek Words via Word Embeddings"_ by V. Barzokas, E. Papagiannopoulou and G. Tsoumakas, published in the proceedings of the [11th Hellenic Conference on Artificial Intelligence (SETN 2020)](https://setn2020.eetn.gr/) and contains the set of tools developed and data prepared for its needs. The paper is going to be available here https://doi.org/10.1145/3411408.3411425 

If you use this code and/or data in your research please cite the following:

```
@inproceedings{Barzokas:2020:SEGWWE:3411408.3411425,
	title 		= {Studying the Evolution of Greek Words via Word Embeddings},
	booktitle 	= {Proceedings of the 11th Hellenic Conference on Artificial Intelligence {SETN} 2020, Athens, Greece, September 02-04, 2020},
	author 		= {Barzokas, Vasileios and 
			   Papagiannopoulou, Eirini and 
			   Tsoumakas, Grigorios},
	publisher 	= {ACM},
	year 		= {2020},
	url 		= {https://doi.org/10.1145/3411408.3411425},
	doi 		= {10.1145/3411408.3411425},
}
```

##### Example visualized result 
![alt text](https://github.com/intelligence-csd-auth-gr/greek-words-evolution/raw/master/assets/results-word-krisi-translated.png "Semantic shift of the word 'crisis'.")

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
#### First steps
* If running for first time, create the text files per period by running:  
    ```shell script
    python gws.py text --action exportByPeriod
    ```

* Then create the models from those text files by running:
    ```shell script
    python gws.py model --action create
    ```

* Later, after the models have been generated you can see the nearest neighbours of a word by running something similar to this example:

    ```shell script
    python gws.py model --action getNN --word ποντίκι --period 2010
    ```
        
    output:
    ```shell script
    ['ποντικι', 'φακα', 'πιασμενο', 'ταμπλετ', 'κατσαβιδι', 'μπιλη', 'γατα', 'ποντικοπαγιδα', 'αραχνη', 'βιντεοκασετα', 'κοριο', 'πληκτρολογιο', 'ποντικο', 'κλακ', 'κατεβασεις', 'μιξερ', 'ποντικακι', 'τσιπακι', 'μεγαλουτσικο', 'συνδεθω', 'μυγοσκοτωστρα']
    ```

* Get the 10 words with the highest semantic change, based on their cosine distance:
        
    ```shell script
    python gws.py model --action getCD --fromPeriod 1980 --toPeriod 2020
    ```
                
* Get the 10 words with the highest semantic change, based on their cosine similarity (opposite sorted list of cosine distance):
        
    ```shell script
    python gws.py model --action getCS --fromPeriod 1980 --toPeriod 2020
    ```

### Options 
The script accepts either of the following positional arguments:
* `website` - allows actions on the websites, such as URL extraction, file downloading etc.
* `metadata` - allows actions on the metadata, metadata display or export etc.
* `text` - allows actions on the text, such as text extraction, metadata display or export etc.
* `model` - allows actions on the trained models, such as the training, evaluation through nearest neighbours or shifts of word meanings through periods.  

In order to see a full list of the available options and a short description of each one of them, type:

    python gws.py --help

The snippets below display a brief description of each of the options that the positional arguments accept.

##### argument: website
```shell script
usage: gws.py website [-h] [--target {openbook}]
                      [--action {fetchLinks,fetchMetadata,fetchFiles}]

optional arguments:
  -h, --help            show this help message and exit
  --target {openbook}   Target website to scrap data from
  --action {fetchLinks,fetchMetadata,fetchFiles}
                        The action to execute on the selected website
```

##### argument: metadata
```shell script
usage: gws.py metadata [-h] [--corpus {all,openbook,project_gutenberg}]
                       [--action {printStandard,printEnhanced,exportEnhanced}]
                       [--fromYear FROMYEAR] [--toYear TOYEAR]
                       [--splitYearsInterval SPLITYEARSINTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  --corpus {all,openbook,project_gutenberg}
                        The name of the target corpus to work with
  --action {printStandard,printEnhanced,exportEnhanced}
                        Action to perform against the metadata of the selected
                        text corpus
  --fromYear FROMYEAR   The target starting year to extract data from
  --toYear TOYEAR       The target ending year to extract data from
  --splitYearsInterval SPLITYEARSINTERVAL
                        The interval to split the years with and export the
                        extracted data
```

##### argument: text
```shell script
usage: gws.py text [-h] [--corpus {all,openbook,project_gutenberg}]
                   [--action {combineByPeriod,extractFromPDF}]
                   [--fromYear FROMYEAR] [--toYear TOYEAR]
                   [--splitYearsInterval SPLITYEARSINTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  --corpus {all,openbook,project_gutenberg}
                        The name of the target corpus to work with
  --action {combineByPeriod,extractFromPDF}
                        Action to perform against the selected text corpus
  --fromYear FROMYEAR   The target starting year to extract data from
  --toYear TOYEAR       The target ending year to extract data from
  --splitYearsInterval SPLITYEARSINTERVAL
                        The interval to split the years with and export the
                        extracted data
```


##### argument: model
```shell script
usage: gws.py model [-h] [--action {create,getNN,getCS,getCD}] [--word WORD]
                    [--period PERIOD] [--textsFolder TEXTSFOLDER]
                    [--fromYear FROMYEAR] [--toYear TOYEAR]

optional arguments:
  -h, --help            show this help message and exit
  --action {create,getNN,getCS,getCD}
                        Action to perform against the selected model
  --word WORD           Target word to get nearest neighbours for
  --period PERIOD       The target period to load the model from
  --textsFolder TEXTSFOLDER
                        The target folder that contains the texts files
  --fromYear FROMYEAR   the target starting year to create the model for
  --toYear TOYEAR       the target ending year to create the model for
```

## License

[Apache License 2.0](LICENSE)

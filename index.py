import warnings
warnings.filterwarnings("ignore")

import argparse
from lib.NearestNeighbours import NearestNeighbours
import glob
import codecs
import os
import re
import fasttext
import pandas
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

DATA_FOLDER = './data'
MODELS_FOLDER = './models'
LIB_FOLDER = './lib'
TEXT_FILES_FOLDER = DATA_FOLDER + '/corpora/project_gutenberg/text'
METADATA_FILENAME = DATA_FOLDER + '/corpora/project_gutenberg/metadata.tsv'
COMBINED_TEXTS_FILENAME = 'corpus_combined.txt'
COMBINED_MODEL_FILENAME = MODELS_FOLDER + '/corpus_combined_model.bin'

FASTTEXT_PATH = './fastText/fasttext'
NEIGHBORS_COUNT = 10

#---------------------------------------------------------------

def readMetadata(filename):
    dataFrame = pandas.read_csv(filename, sep='\t')

    dataFrame['text'] = ''
    dataFrame['publishedYear'] = ''

    return dataFrame

def getTextFileContents(filename):
    with codecs.open(filename, 'r', 'utf-8') as file:
        fileContents = file.read().splitlines()

    file.close()

    return fileContents

def extractPublishedYear(text):
    # a match of a year in the form of "1 8 2 1", as the first match after the common publishing names
    match1 = re.findall('(?:ΑΘΗΝΑ|ΑΘΗΝΑΙ|ΑΘΗΝΗΣΙ|ΠΕΙΡΑΙΕΥΣ|ΖΑΚΥΝΘΩ|ΣΜΥΡΝΗ|ΠΑΡΙΣΙΟΙΣ)(?:[\s\D]\d{0,3})*(\d+(?:\s+\d+){3})', text, re.MULTILINE | re.IGNORECASE)
    # a match in the form of "1821", as the first match after the common publishing names
    match2 = re.findall('(?:ΑΘΗΝΑ|ΑΘΗΝΑΙ|ΑΘΗΝΗΣΙ|ΠΕΙΡΑΙΕΥΣ|ΖΑΚΥΝΘΩ|ΣΜΥΡΝΗ|ΠΑΡΙΣΙΟΙΣ)(?:[\s\D]\d{0,3})*(\d{4})', text, re.MULTILINE | re.IGNORECASE)
    # a match in the form of "1821", as the first match in the whole provided text
    match3 = re.findall('(\d{4})', text, re.MULTILINE | re.IGNORECASE)

    if match1:
        return int(match1[0].replace(' ', ''))
    elif match2:
        return int(match2[0])
    elif match3:
        return int(match3[0])

    return 0

def estimatePublishedYear(authorYearOfBirth, authorYearOfDeath):
    estimatedPublishedYear = ''

    if authorYearOfBirth > 1800:
        estimatedPublishedYear = (authorYearOfBirth + authorYearOfDeath) // 2

    return estimatedPublishedYear

def enhanceMetadata(metadata):
    for index, row in metadata.iterrows():
        textFileContentsAsLines = getTextFileContents(TEXT_FILES_FOLDER + '/' + row['id'] + '.txt')
        textFileContentsAsString = " ".join(textFileContentsAsLines)

        metadata.loc[index, 'text'] = textFileContentsAsString
        firstNLines = " ".join(textFileContentsAsLines[:100])

        publishedYear = extractPublishedYear(firstNLines) or \
                        estimatePublishedYear(row['authorYearOfBirth'], row['authorYearOfDeath'])

        metadata.loc[index, 'publishedYear'] = publishedYear or -1

        #print(row['id'] + ' - %s' % publishedYear)

        #metadata.loc[index, 'tokensCount'] = int(len(nltk.word_tokenize(textFileContentsAsString)))

    return metadata

def exportTextToFile(text, filename):
    f = open(filename, 'w')
    f.write(text)
    f.close()

def exportMetadata(metadata, filename):
    metadata.to_csv(filename, sep='\t', index=False, header=True, columns=['id', 'title', 'author',
                                                                           'authorYearOfBirth',
                                                                           'authorYearOfDeath',
                                                                           'publishedYear',
                                                                           'tokensCount'])

def exportTextByDecade(enhancedMetadata):
    for i in range(1800, 2000, 10):
        text = enhancedMetadata.loc[(enhancedMetadata['publishedYear'] >= i) &
                                    (enhancedMetadata['publishedYear'] < (i + 10))]
        #print(text)
        exportTextToFile(text['text'].str.cat(sep='\n'), './tmp/%s.txt' % i)

def createModel(textsFilename, modelsFolder):
    model = fasttext.train_unsupervised(textsFilename, model='skipgram')
    modelFilename = os.path.join(modelsFolder, os.path.basename(textsFilename).replace('txt', 'model'))
    model.save_model(modelFilename)

def createModelsFromTextFiles(textFilesFolder):
    filenames = glob.glob(textFilesFolder + '/*.txt')

    for filename in filenames:
        if os.stat(filename).st_size != 0:
            createModel(filename, MODELS_FOLDER)

#---------------------------------------------------------------

def metadataParser(args):
    metadata = readMetadata(METADATA_FILENAME)
    enhancedMetadata = enhanceMetadata(metadata)

    if args.printStandard:
        print(metadata)
    if args.printEnhanced:
        print(enhancedMetadata)
    if args.export:
        exportMetadata(enhancedMetadata, METADATA_FILENAME)
    if exportTextByDecade:
        exportTextByDecade(enhancedMetadata)

def modelParser(args):
    if args.action == 'create':
        createModelsFromTextFiles(args.textsFolder)
    elif args.action == 'getNN':
        nearestNeighbours = NearestNeighbours(FASTTEXT_PATH, './models/' + args.decade + '.model', NEIGHBORS_COUNT)
        print(nearestNeighbours.getNeighboursForWord(args.word))

parser = argparse.ArgumentParser()
parser.add_argument('--version', action='version', version='0.0.1')
subparsers = parser.add_subparsers()

parser_metadata = subparsers.add_parser('metadata')
parser_metadata.add_argument('--printStandard', action='store_true', help='print the standard metadata')
parser_metadata.add_argument('--printEnhanced', action='store_true', help='print the enhanced metadata')
parser_metadata.add_argument('--export', action='store_true', help='export the enhanced metadata')
parser_metadata.add_argument('--exportTextByDecade', action='store_true', help='export the text by decade')
parser_metadata.set_defaults(func=metadataParser)

parser_model = subparsers.add_parser('model')
parser_model.add_argument('--action', default='getNN', help='target action.')
parser_model.add_argument('--word', help='target word to get nearest neighbours for.')
parser_model.add_argument('--decade', help='the target decade to load the model from.')
parser_model.add_argument('--textsFolder', default='./tmp', help='the target folder that contains the texts files.')
parser_model.set_defaults(func=modelParser)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)
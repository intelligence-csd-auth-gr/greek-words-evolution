import warnings
import argparse
import fasttext
import os
import pandas
import logging
from lib.file import readMetadata, exportTextToFile
from lib.text import enhanceMetadata, exportTextByPeriod, exportMetadata, preProcessText
from lib.model import createModelsFromTextFiles, getNeighboursForWord
from lib.vector import getCosineDistance, getCosineSimilarity


warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################
DATA_FOLDER = os.path.join(os.path.curdir, 'data')
MODELS_FOLDER = os.path.join(os.path.curdir, 'output', 'models')
PRODUCED_TEXTS_FOLDER = os.path.join(os.path.curdir, 'output', 'texts')
LIB_FOLDER = os.path.join(os.path.curdir, 'lib')
MODEL_FILE_EXTENSION = '.model'
TEXT_FILE_EXTENSION = '.txt'
CORPORA = [
    {
        'name': 'openbook',
        'textFilesFolder': os.path.join(DATA_FOLDER, 'corpora', 'openbook', 'text', 'parsable'),
        'metadataFilename': os.path.join(DATA_FOLDER, 'corpora', 'openbook', 'metadata.tsv')
    },
    {
        'name': 'project_gutenberg',
        'textFilesFolder': os.path.join(DATA_FOLDER, 'corpora', 'project_gutenberg', 'text', 'parsable'),
        'metadataFilename': os.path.join(DATA_FOLDER, 'corpora', 'project_gutenberg', 'metadata.tsv')
    },
]
COMBINED_TEXTS_FILENAME = 'corpus_combined.txt'
COMBINED_MODEL_FILENAME = os.path.join(MODELS_FOLDER, 'corpus_combined_model.bin')
NEIGHBORS_COUNT = 20

########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################


def metadataParser(args):
    metadataList = []
    enhancedMetadataList = []

    for corpus in CORPORA:
        if (not args.corpusName) or (args.corpusName and corpus['name'] == args.corpusName):
            textFilesFolder = corpus.get('textFilesFolder')
            metadata = readMetadata(corpus.get('metadataFilename'))
            metadataList.append(metadata)

            if args.printEnhanced or args.exportEnhancedMetadata or args.exportTextByPeriod:
                enhancedMetadata = enhanceMetadata(textFilesFolder, metadata=metadata, detectPublishedYear=False,
                                                   calculateTokens=False)

                enhancedMetadataList.append(enhancedMetadata)

    if args.printStandard:
        combinedMetadata = pandas.concat(metadataList)
        print(combinedMetadata)
    elif args.printEnhanced or args.exportEnhancedMetadata or args.exportTextByPeriod:
        combinedEnhancedMetadata = pandas.concat(enhancedMetadataList)

        if args.printEnhanced:
            print(combinedEnhancedMetadata)
        if args.exportEnhancedMetadata:
            exportMetadata(combinedEnhancedMetadata)
        if args.exportTextByPeriod:
            exportTextByPeriod(combinedEnhancedMetadata, args.fromYear, args.toYear, args.splitYearsInterval)


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################


def modelParser(args):
    if args.action == 'create':
        logger.info('Selected action: Create models')

        createModelsFromTextFiles(args.textsFolder)
    elif args.action == 'getNN':
        logger.info('Selected action: Retrieve nearest neighbours')

        modelFilename = args.period + MODEL_FILE_EXTENSION
        print(getNeighboursForWord(preProcessText(args.word), modelFilename, NEIGHBORS_COUNT))
    elif args.action == 'getCD' or args.action == 'getCS':
        if args.action == 'getCD':
            logger.info('Selected action: Get cosine distance')
        elif args.action == 'getCS':
            logger.info('Selected action: Get cosine similarity')

        fromYearFilename = args.fromYear + MODEL_FILE_EXTENSION
        toYearFilename = args.toYear + MODEL_FILE_EXTENSION

        modelA = fasttext.load_model(os.path.join(MODELS_FOLDER, fromYearFilename))
        modelB = fasttext.load_model(os.path.join(MODELS_FOLDER, toYearFilename))

        clearVectorModelA = {}
        clearVectorModelB = {}

        for label in modelA.get_labels():
            clearVectorModelA[label] = modelA.get_word_vector(label)

        for label in modelB.get_labels():
            clearVectorModelB[label] = modelB.get_word_vector(label)

        # alignedEmbeddingsB = align_two_embeddings(clearVectorModelA, clearVectorModelB)

        results = {}
        for word in modelA.words:
            if word in modelB.words:
                if args.action == 'getCD':
                    results[word] = getCosineDistance(clearVectorModelA[word], clearVectorModelB[word])
                elif args.action == 'getCS':
                    results[word] = getCosineSimilarity(clearVectorModelA[word], clearVectorModelB[word])

        if args.action == 'getCD':
            sortedResults = sorted(results.items(), key=lambda x: x[1], reverse=True)
        elif args.action == 'getCS':
            sortedResults = sorted(results.items(), key=lambda x: x[1])

        resultsPerPeriod = {}

        for wordTuple in sortedResults[:50]:
            word = wordTuple[0]
            resultsPerPeriod[word] = {}

            resultsPerPeriod[word][str(args.fromYear)] = getNeighboursForWord(word, fromYearFilename, NEIGHBORS_COUNT)
            resultsPerPeriod[word][str(args.toYear)] = getNeighboursForWord(word, toYearFilename, NEIGHBORS_COUNT)

        # print(resultsPerPeriod)
        exportTextToFile(resultsPerPeriod, './shifts.json', True)

########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--version', action='version', version='0.0.3')
subparsers = parser.add_subparsers()

parser_metadata = subparsers.add_parser('metadata')
parser_metadata.add_argument('--corpusName', help='The name of the target corpus to work with')
parser_metadata.add_argument('--printStandard', action='store_true', help='Prints the standard metadata')
parser_metadata.add_argument('--printEnhanced', action='store_true', help='Prints the enhanced metadata')
parser_metadata.add_argument('--exportEnhancedMetadata', action='store_true', help='Exports the enhanced metadata')
parser_metadata.add_argument('--exportTextByPeriod', action='store_true', help='Exports the text by period')
parser_metadata.add_argument('--fromYear', default=1800, type=int, help='The target starting year to extract data from')
parser_metadata.add_argument('--toYear', default=1900, type=int, help='The target ending year to extract data from')
parser_metadata.add_argument('--splitYearsInterval', default=10, type=int, help='The interval to split the years with '
                                                                                'and export the extracted data')
parser_metadata.set_defaults(func=metadataParser)

parser_model = subparsers.add_parser('model')
parser_model.add_argument('--action', default='getNN', choices=['getNN', 'getCS', 'getCD'], help='Action to perform against the selected model')
parser_model.add_argument('--word', help='Target word to get nearest neighbours for')
parser_model.add_argument('--period', help='The target period to load the model from')
parser_model.add_argument('--textsFolder', default='./output/texts', help='The target folder that contains the '
                                                                          'texts files')
parser_model.add_argument('--fromYear', default='1800', help='the target starting year to create the model for.')
parser_model.add_argument('--toYear', default='1900', help='the target ending year to create the model for.')
parser_model.set_defaults(func=modelParser)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)

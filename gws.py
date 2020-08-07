import warnings
import argparse
import os
import logging
import lib.metadata as metadata
import lib.model as model
import lib.text as text
import lib.website as website


warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################
DATA_FOLDER = os.path.join(os.path.curdir, 'data')
MODELS_FOLDER = os.path.join(os.path.curdir, 'output', 'models')
SCRAPPED_PDF_FOLDER = os.path.join(os.path.curdir, 'data', 'scrap', 'pdf')
FASTTEXT_PATH = os.path.join(os.path.curdir, 'fastText', 'fasttext')
SCRAPPED_TEXT_FOLDER = os.path.join(os.path.curdir, 'data', 'scrap', 'text')
PRODUCED_TEXTS_FOLDER = os.path.join(os.path.curdir, 'output', 'texts')
LIB_FOLDER = os.path.join(os.path.curdir, 'lib')
MODEL_FILE_EXTENSION = '.model'
TEXT_FILE_EXTENSION = '.txt'
PDF_FILE_EXTENSION = '.pdf'
POST_URLS_FILENAME = 'post_urls.pickle'
METADATA_FILENAME = 'raw_metadata.csv'
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

#####################################
# Set up required folders and perform any other preliminary tasks
#####################################
if not os.path.exists(SCRAPPED_PDF_FOLDER):
    os.makedirs(SCRAPPED_PDF_FOLDER)

if not os.path.exists(SCRAPPED_TEXT_FOLDER):
    os.makedirs(SCRAPPED_TEXT_FOLDER)

########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################


def websiteParser(args):
    if args.action == 'fetchLinks':
        logger.info('Selected action: Fetch website links')

        links = website.fetchLinks(args.target)
        print(links)
    elif args.action == 'fetchMetadata':
        logger.info('Selected action: Fetch website metadata')

        metadata = website.fetchMetadata(args.target, PDF_FILE_EXTENSION, METADATA_FILENAME)
        print(metadata)
    elif args.action == 'fetchFiles':
        logger.info('Selected action: Fetch website files')

        website.fetchFiles(args.target, PDF_FILE_EXTENSION, METADATA_FILENAME, SCRAPPED_PDF_FOLDER)

########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################


def metadataParser(args):
    if (args.action == 'printStandard'):
        combinedMetadata = metadata.getCombined(CORPORA, args.corpus, False)
        print(combinedMetadata)
    elif (args.action == 'printEnhanced' or args.action == 'exportEnhanced'):
        combinedMetadata = metadata.getCombined(CORPORA, args.corpus, True)

        if args.action == 'printEnhanced':
            print(combinedMetadata)
        if args.action == 'exportEnhanced':
            text.exportMetadata(combinedMetadata)


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################


def textParser(args):
    combinedMetadata = metadata.getCombined(CORPORA, args.corpus, True)

    if args.action == 'exportByPeriod':
        logger.info('Selected action: Export combined text by period')

        text.exportTextByPeriod(combinedMetadata, args.fromYear, args.toYear, args.splitYearsInterval)

    elif args.action == 'extractFromPDF':
        logger.info('Selected action: Extract text from PDF')

        text.extractTextFromPdf(combinedMetadata, SCRAPPED_PDF_FOLDER, PDF_FILE_EXTENSION, SCRAPPED_TEXT_FOLDER,
                                TEXT_FILE_EXTENSION)

########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################


def modelParser(args):
    if args.action == 'create':
        logger.info('Selected action: Create models')

        model.createModelsFromTextFiles(args.textsFolder, TEXT_FILE_EXTENSION, MODELS_FOLDER, MODEL_FILE_EXTENSION)
    elif args.action == 'getNN':
        logger.info('Selected action: Retrieve Nearest Neighbours')

        modelFilename = args.period + MODEL_FILE_EXTENSION
        nearestNeighbours = model.getNeighboursForWord(text.preProcessText(args.word), modelFilename, MODELS_FOLDER,
                                                       FASTTEXT_PATH, NEIGHBORS_COUNT)

        print(nearestNeighbours)
    elif args.action == 'getCD':
        logger.info('Selected action: Get cosine distance')

        model.exportByDistance(args.action, MODEL_FILE_EXTENSION, MODELS_FOLDER, args.fromYear, args.toYear,
                               NEIGHBORS_COUNT, FASTTEXT_PATH)
    elif args.action == 'getCS':
        logger.info('Selected action: Get cosine similarity')

        model.exportByDistance(args.action, MODEL_FILE_EXTENSION, MODELS_FOLDER, args.fromYear, args.toYear,
                               NEIGHBORS_COUNT, FASTTEXT_PATH)

########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--version', action='version', version='1.0.0')
subparsers = parser.add_subparsers()


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################


parser_website = subparsers.add_parser('website')
parser_website.add_argument('--target', default='openbook', choices=['openbook'], help='Target website to '
                                                                                       'scrap data from')
parser_website.add_argument('--action', default='fetchFiles', choices=['fetchLinks', 'fetchMetadata', 'fetchFiles'],
                            help='The action to execute on the selected website')
parser_website.set_defaults(func=websiteParser)


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################


parser_metadata = subparsers.add_parser('metadata')
parser_metadata.add_argument('--corpus', default='all', choices=['all', 'openbook', 'project_gutenberg'],
                             help='The name of the target corpus to work with')
parser_metadata.add_argument('--action', default='printStandard', choices=['printStandard', 'printEnhanced',
                                                                           'exportEnhanced'],
                             help='Action to perform against the metadata of the selected text corpus')
parser_metadata.add_argument('--fromYear', default=1800, type=int, help='The target starting year to extract data from')
parser_metadata.add_argument('--toYear', default=1900, type=int, help='The target ending year to extract data from')
parser_metadata.add_argument('--splitYearsInterval', default=10, type=int, help='The interval to split the years with '
                                                                                'and export the extracted data')
parser_metadata.set_defaults(func=metadataParser)

########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################

parser_text = subparsers.add_parser('text')
parser_text.add_argument('--corpus', default='all', choices=['all', 'openbook', 'project_gutenberg'],
                         help='The name of the target corpus to work with')
parser_text.add_argument('--action', default='exportByPeriod', choices=['exportByPeriod', 'extractFromPDF'],
                         help='Action to perform against the selected text corpus')
parser_text.add_argument('--fromYear', default=1800, type=int, help='The target starting year to extract data from')
parser_text.add_argument('--toYear', default=1900, type=int, help='The target ending year to extract data from')
parser_text.add_argument('--splitYearsInterval', default=10, type=int, help='The interval to split the years with '
                                                                            'and export the extracted data')
parser_text.set_defaults(func=textParser)

########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################

parser_model = subparsers.add_parser('model')
parser_model.add_argument('--action', default='getNN', choices=['create', 'getNN', 'getCS', 'getCD'],
                          help='Action to perform against the selected model')
parser_model.add_argument('--word', help='Target word to get nearest neighbours for')
parser_model.add_argument('--period', help='The target period to load the model from')
parser_model.add_argument('--textsFolder', default='./output/texts', help='The target folder that contains the '
                                                                          'texts files')
parser_model.add_argument('--fromYear', default='1800', help='the target starting year to create the model for')
parser_model.add_argument('--toYear', default='1900', help='the target ending year to create the model for')
parser_model.set_defaults(func=modelParser)

########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)

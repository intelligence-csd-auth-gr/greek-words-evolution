import warnings
import argparse
import codecs
import fasttext
import glob
import nltk
import time
import os
import pandas
import re
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging
from lib.NearestNeighbours import NearestNeighbours
# from lib.align_embeddings import align_two_embeddings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################

DATA_FOLDER = './data'
MODELS_FOLDER = './output/models'
PRODUCED_TEXTS_FOLDER = './output/texts'
LIB_FOLDER = './lib'
MODEL_FILE_EXTENSION = '.model'
TEXT_FILE_EXTENSION = '.txt'
CORPORA = [
    {
        'name': 'openbook',
        'textFilesFolder': DATA_FOLDER + '/corpora/openbook/text/parsable',
        'metadataFilename': DATA_FOLDER + '/corpora/openbook/metadata.tsv'
    },
    {
        'name': 'project_gutenberg',
        'textFilesFolder': DATA_FOLDER + '/corpora/project_gutenberg/text/parsable',
        'metadataFilename': DATA_FOLDER + '/corpora/project_gutenberg/metadata.tsv'
    },
]
COMBINED_TEXTS_FILENAME = 'corpus_combined.txt'
COMBINED_MODEL_FILENAME = MODELS_FOLDER + '/corpus_combined_model.bin'
FASTTEXT_PATH = './fastText/fasttext'
NEIGHBORS_COUNT = 20

########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################


def readMetadata(filename):
    logger.info('Reading metadata from %s', filename)

    dataFrame = pandas.read_csv(filename, sep='\t')

    dataFrame['text'] = ''

    return dataFrame


def getTextFileContents(filename):
    if os.path.isfile(filename):
        with codecs.open(filename, 'r', 'utf-8') as file:
            fileContents = file.read().splitlines()
        file.close()
    else:
        fileContents = []

    return fileContents


def extractPublishedYear(text):
    # a match of a year in the form of "1 8 2 1", as the first match after the common publishing names
    match1 = re.findall(
        r'(?:ΑΘΗΝΑ|ΑΘΗΝΑΙ|ΑΘΗΝΗΣΙ|ΠΕΙΡΑΙΕΥΣ|ΖΑΚΥΝΘΩ|ΣΜΥΡΝΗ|ΠΑΡΙΣΙΟΙΣ)(?:[\s\D]\d{0,3})*(\d+(?:\s+\d+){3})', text,
        re.MULTILINE | re.IGNORECASE)
    # a match in the form of "1821", as the first match after the common publishing names
    match2 = re.findall(
        r'(?:ΑΘΗΝΑ|ΑΘΗΝΑΙ|ΑΘΗΝΗΣΙ|ΠΕΙΡΑΙΕΥΣ|ΖΑΚΥΝΘΩ|ΣΜΥΡΝΗ|ΠΑΡΙΣΙΟΙΣ)(?:[\s\D]\d{0,3})*(\d{4})', text,
        re.MULTILINE | re.IGNORECASE)
    # a match in the form of "1821", as the first match in the whole provided text
    match3 = re.findall(r'(\d{4})', text, re.MULTILINE | re.IGNORECASE)

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


def enhanceMetadata(textFilesFolder, metadata, detectPublishedYear, calculateTokens):
    logger.info('Enhancing metadata')

    for index, row in metadata.iterrows():
        textFileContentsAsLines = getTextFileContents(os.path.join(textFilesFolder, row['id'] + TEXT_FILE_EXTENSION))
        textFileContentsAsString = '\n'.join(textFileContentsAsLines)

        metadata.loc[index, 'text'] = textFileContentsAsString

        if detectPublishedYear:
            firstNLines = " ".join(textFileContentsAsLines[:100])

            publishedYear = extractPublishedYear(firstNLines) or estimatePublishedYear(row['authorYearOfBirth'],
                                                                                       row['authorYearOfDeath'])

            metadata.loc[index, 'publishedYear'] = publishedYear or -1

        if calculateTokens:
            metadata.loc[index, 'tokensCount'] = int(len(nltk.word_tokenize(textFileContentsAsString)))

    return metadata


def preProcessText(text):
    logger.info('Preprocessing text')

    stopWords = set(nltk.corpus.stopwords.words('greek'))

    text = text.lower()

    # move split words into same line. The second - is a different utf character than the first,
    # which is the normal dash
    text = re.sub(r'[-­]\s+', '', text)

    # remove stopwords
    text = re.compile(r'\b(' + r'|'.join(stopWords) + r')\b\s*').sub('', text)

    # remove special characters
    # text = re.sub('[»«‒§—·•■·][^A-Za-z0-9]', '', text)

    # remove anything that is not latin or greek letters
    text = re.sub(r'[^Α-Ωα-ωΊίϊΐΌόΆάΈέΎύϋΰΉήΏώ\s]', '', text)

    # remove all accents from vowels
    text = re.sub('[άἀἁἂἃἄἅἆἇὰάᾀᾁᾂᾃᾄᾅᾆᾇᾰᾱᾲᾳᾴᾶᾷ]', 'α', text)
    text = re.sub('[ΆἈἉἊἋἌἍἎἏᾈᾉᾊᾋᾌᾍᾎᾏᾸᾹᾺΆᾼ]', 'Α', text)
    text = re.sub('[έἐἑἒἓἔἕὲέ]', 'ε', text)
    text = re.sub('[ΈἙἚἛἜἝ]', 'Ε', text)
    text = re.sub('[ήἠἡἢἣἤἥἦἧῆὴῇ]', 'η', text)
    text = re.sub('[ΉἨἩἪἫἬἭἮἯ]', 'Η', text)
    text = re.sub('[ίἰἱἲἳἴἵὶῖ]', 'ι', text)
    text = re.sub('[ΊἶἷἸἹἺἻἼἽἾἿ]', 'Ι', text)
    text = re.sub('[όὀὁὂὃὄὅὸ]', 'ο', text)
    text = re.sub('[ΌὈὉὊὋὌὍ]', 'Ο', text)
    text = re.sub('[ύὐὑὒὓὔὕὖὗ]', 'υ', text)
    text = re.sub('[ΎὙὛὝὟ]', 'Υ', text)
    text = re.sub('[ώὠὡὢὣὤὥὦὧῶ]', 'ω', text)
    text = re.sub('[ΏὨὩὪὫὬὭὮὯ]', 'Ω', text)

    # remove single character words
    text = re.sub(r'\b[α-ωΑ-Ω]\b', '', text)

    # remove digits
    # text = re.sub(r'\d+', '', text)

    # remove multiple whitespaces
    text = re.sub(r'\s\s+', ' ', text)

    # remove punctuation
    # text = re.compile('[%s]' % re.escape(string.punctuation)).sub('', text)

    return text


def exportTextToFile(text, filename, exportToJson=False):
    fileHandler = open(filename, 'w', encoding='utf8')

    if exportToJson:
        json.dump(text, fileHandler, indent=4, ensure_ascii=False)
    else:
        fileHandler.write(text)

    fileHandler.close()


def exportMetadata(metadata, filename='-export.tsv'):
    exportFilename = int(time.time()) + filename

    metadata.to_csv(exportFilename, sep='\t', index=False, header=True, columns=['id', 'title', 'author',
                                                                                 'authorYearOfBirth',
                                                                                 'authorYearOfDeath',
                                                                                 'publishedYear',
                                                                                 'tokensCount'])


def exportTextByPeriod(enhancedMetadata, fromYear, toYear, splitYearsInterval):
    for i in range(fromYear, toYear, splitYearsInterval):
        currentRangeFrom = i
        currentRangeTo = i + splitYearsInterval

        logger.info('Exporting text from ' + str(currentRangeFrom) + ' to ' + str(currentRangeTo))

        text = enhancedMetadata.loc[(enhancedMetadata['publishedYear'] >= currentRangeFrom) &
                                    (enhancedMetadata['publishedYear'] < currentRangeTo)]
        preProcessedText = preProcessText(text['text'].str.cat(sep='\n'))
        exportTextToFile(preProcessedText, os.path.join(PRODUCED_TEXTS_FOLDER, ('%s' + TEXT_FILE_EXTENSION) % i))


def createModel(textsFilename, modelsFolder):
    logger.info('Creating model from %s', textsFilename)

    # model             # unsupervised fasttext model {cbow, skipgram} [skipgram]
    # lr                # learning rate [0.05]
    # dim               # size of word vectors [100]
    # ws                # size of the context window [5]
    # epoch             # number of epochs [5]
    # minCount          # minimal number of word occurences [5]
    # minn              # min length of char ngram [3]
    # maxn              # max length of char ngram [6]
    # neg               # number of negatives sampled [5]
    # wordNgrams        # max length of word ngram [1]
    # loss              # loss function {ns, hs, softmax, ova} [ns]
    # bucket            # number of buckets [2000000]
    # thread            # number of threads [number of cpus]
    # lrUpdateRate      # change the rate of updates for the learning rate [100]
    # t                 # sampling threshold [0.0001]
    model = fasttext.train_unsupervised(textsFilename, ws=20, model='skipgram', epoch=5, dim=100, minCount=15, minn=3,
                                        maxn=6, neg=5, thread=12)
    modelFilename = os.path.join(modelsFolder, os.path.basename(textsFilename).replace(TEXT_FILE_EXTENSION,
                                                                                       MODEL_FILE_EXTENSION))
    model.save_model(modelFilename)


def createModelsFromTextFiles(textFilesFolder):
    filenames = glob.glob(os.path.join(textFilesFolder, '*' + TEXT_FILE_EXTENSION))

    for filename in filenames:
        if os.stat(filename).st_size != 0:
            with open(filename, 'r') as file:
                contents = file.read().strip()
                file.close()

                if len(contents) != 0:
                    createModel(filename, MODELS_FOLDER)
                else:
                    logger.info('File %s is empty. Skipping the model creation.', filename)


def getCosineDistanceOfVectors(vectorA, vectorB):
    return spatial.distance.cosine(vectorA, vectorB)


def getCosineSimilarityOfVectors(vectorA, vectorB):
    # assuming that the two vectors have already the same size
    size = len(vectorA)

    return cosine_similarity(vectorA.reshape(1, size), vectorB.reshape(1, size))[0][0]

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
            exportTextByPeriod(combinedEnhancedMetadata)


def modelParser(args):
    if args.action == 'create':
        logger.info('Selected action: Create models')

        createModelsFromTextFiles(args.textsFolder)
    elif args.action == 'getNN':
        logger.info('Selected action: Retrieve nearest neighbours')

        nearestNeighbours = NearestNeighbours(FASTTEXT_PATH,
                                              os.path.join(MODELS_FOLDER, args.period + MODEL_FILE_EXTENSION),
                                              NEIGHBORS_COUNT)

        print(nearestNeighbours.getNeighboursForWord(preProcessText(args.word)))
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
                    results[word] = getCosineDistanceOfVectors(clearVectorModelA[word], clearVectorModelB[word])
                elif args.action == 'getCS':
                    results[word] = getCosineSimilarityOfVectors(clearVectorModelA[word], clearVectorModelB[word])

        if args.action == 'getCD':
            sortedResults = sorted(results.items(), key=lambda x: x[1], reverse=True)
        elif args.action == 'getCS':
            sortedResults = sorted(results.items(), key=lambda x: x[1])

        resultsPerPeriod = {}

        for wordTuple in sortedResults[:50]:
            word = wordTuple[0]
            resultsPerPeriod[word] = {}

            nearestNeighbours = NearestNeighbours(FASTTEXT_PATH,
                                                  os.path.join(MODELS_FOLDER, fromYearFilename),
                                                  NEIGHBORS_COUNT)

            nn = nearestNeighbours.getNeighboursForWord(word)
            resultsPerPeriod[word][str(args.fromYear)] = nn

            nearestNeighbours = NearestNeighbours(FASTTEXT_PATH,
                                                  os.path.join(MODELS_FOLDER, toYearFilename),
                                                  NEIGHBORS_COUNT)

            nn = nearestNeighbours.getNeighboursForWord(word)
            resultsPerPeriod[word][str(args.toPeriod)] = nn

        print(resultsPerPeriod)
        exportTextToFile(resultsPerPeriod, './shifts.json', True)

########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--version', action='version', version='0.0.2')
subparsers = parser.add_subparsers()

parser_metadata = subparsers.add_parser('metadata')
parser_metadata.add_argument('--corpusName', help='The name of the target corpus to work with')
parser_metadata.add_argument('--printStandard', action='store_true', help='Prints the standard metadata')
parser_metadata.add_argument('--printEnhanced', action='store_true', help='Prints the enhanced metadata')
parser_metadata.add_argument('--exportEnhancedMetadata', action='store_true', help='Exports the enhanced metadata')
parser_metadata.add_argument('--exportTextByPeriod', action='store_true', help='Exports the text by period')
parser_metadata.add_argument('--fromYear', default='1800', help='The target starting year to extract data from')
parser_metadata.add_argument('--toYear', default='1900', help='The target ending year to extract data from')
parser_metadata.add_argument('--splitYearsInterval', default='10', help='The interval to split the years with '
                                                                        'and export the extracted data')
parser_metadata.set_defaults(func=metadataParser)

parser_model = subparsers.add_parser('model')
parser_model.add_argument('--action', default='getNN', help='Action to perform against the selected model')
parser_model.add_argument('--word', help='Target word to get nearest neighbours for')
parser_model.add_argument('--period', help='The target period to load the model from')
parser_model.add_argument('--textsFolder', default='./output/texts', help='The target folder that contains the '
                                                                          'texts files')
parser_model.add_argument('--fromPeriod', default='1800', help='the target starting period.')
parser_model.add_argument('--toPeriod', default='1900', help='the target ending period.')
parser_model.set_defaults(func=modelParser)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)

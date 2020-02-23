import warnings
warnings.filterwarnings('ignore')

import argparse
import codecs
import fasttext
import glob
from lib.NearestNeighbours import NearestNeighbours
import nltk
import os
import pandas
import re
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------------------------------------------------
########################################################################################################################

DATA_FOLDER = './data'
MODELS_FOLDER = './output/models'
PRODUCED_TEXTS_FOLDER = './output/texts'
LIB_FOLDER = './lib'
TEXT_FILES_FOLDER = DATA_FOLDER + '/corpora/openbook/text/parsable'
MODEL_FILE_EXTENSION = '.model'
TEXT_FILE_EXTENSION = '.txt'
METADATA_FILENAME = DATA_FOLDER + '/corpora/openbook/metadata.tsv'
COMBINED_TEXTS_FILENAME = 'corpus_combined.txt'
COMBINED_MODEL_FILENAME = MODELS_FOLDER + '/corpus_combined_model.bin'
START_YEAR = 1900
END_YEAR = 2020
SPLIT_YEARS_INTERVAL = 10

FASTTEXT_PATH = './fastText/fasttext'
NEIGHBORS_COUNT = 20

########################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------
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

def enhanceMetadata(metadata, detectPublishedYear, calculateTokens):
    logger.info('Enhancing metadata')

    for index, row in metadata.iterrows():
        textFileContentsAsLines = getTextFileContents(os.path.join(TEXT_FILES_FOLDER, row['id'] + TEXT_FILE_EXTENSION))
        textFileContentsAsString = preProcessText('\n'.join(textFileContentsAsLines))

        metadata.loc[index, 'text'] = textFileContentsAsString

        if detectPublishedYear:
            firstNLines = " ".join(textFileContentsAsLines[:100])

            publishedYear = extractPublishedYear(firstNLines) or \
                            estimatePublishedYear(row['authorYearOfBirth'], row['authorYearOfDeath'])

            metadata.loc[index, 'publishedYear'] = publishedYear or -1

        if calculateTokens:
            metadata.loc[index, 'tokensCount'] = int(len(nltk.word_tokenize(textFileContentsAsString)))

    return metadata

def preProcessText(text):
    stopWords = set(nltk.corpus.stopwords.words('greek'))

    text = text.lower()
    text = re.sub('-\n', '', text)
    # remove special characters
    #text = re.sub('[»«‒§—·•■·][^A-Za-z0-9]', '', text)
    # remove stopwords
    text = re.compile(r'\b(' + r'|'.join(stopWords) + r')\b\s*').sub('', text)
    # remove anything that is not latin or greek letters
    text = re.sub('[^Α-Ωα-ωΊίϊΐΌόΆάΈέΎύϋΰΉήΏώ\s]', '', text)
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
    #text = re.sub(r'\d+', '', text)
    # remove multiple whitespaces
    text = re.sub('\s\s+', ' ', text)
    # remove punctuation
    #text = re.compile('[%s]' % re.escape(string.punctuation)).sub('', text)

    return text

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

def exportTextByPeriod(enhancedMetadata):
    for i in range(START_YEAR, END_YEAR, SPLIT_YEARS_INTERVAL):
        currentRangeFrom = i
        currentRangeTo = i + SPLIT_YEARS_INTERVAL

        logger.info('Exporting text from ' + str(currentRangeFrom) + ' to ' + str(currentRangeTo))

        text = enhancedMetadata.loc[(enhancedMetadata['publishedYear'] >= currentRangeFrom) &
                                    (enhancedMetadata['publishedYear'] < currentRangeTo)]
        exportTextToFile(text['text'].str.cat(sep='\n'), os.path.join(PRODUCED_TEXTS_FOLDER, ('%s' + TEXT_FILE_EXTENSION) % i))

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
    model = fasttext.train_unsupervised(textsFilename, model='skipgram', epoch=5, dim=100, minCount=3, minn=0, maxn=0, neg=10, thread=12)
    modelFilename = os.path.join(modelsFolder, os.path.basename(textsFilename).replace(TEXT_FILE_EXTENSION, MODEL_FILE_EXTENSION))
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
#-----------------------------------------------------------------------------------------------------------------------
########################################################################################################################

def metadataParser(args):
    metadata = readMetadata(METADATA_FILENAME)
    enhancedMetadata = enhanceMetadata(metadata=metadata, detectPublishedYear=False, calculateTokens=False)

    if args.printStandard:
        print(metadata)
    if args.printEnhanced:
        print(enhancedMetadata)
    if args.export:
        exportMetadata(enhancedMetadata, METADATA_FILENAME)
    if args.exportTextByPeriod:
        exportTextByPeriod(enhancedMetadata)

def modelParser(args):
    if args.action == 'create':
        logger.info('Selected action: Create models')

        createModelsFromTextFiles(args.textsFolder)
    elif args.action == 'getNN':
        logger.info('Selected action: Retrieve nearest neighbours')

        nearestNeighbours = NearestNeighbours(FASTTEXT_PATH, os.path.join(MODELS_FOLDER, args.period + MODEL_FILE_EXTENSION), NEIGHBORS_COUNT)

        print(nearestNeighbours.getNeighboursForWord(preProcessText(args.word)))
    elif args.action == 'getCD':
        logger.info('Selected action: Get cosine distance')

        fromPeriodFilename = args.fromPeriod + MODEL_FILE_EXTENSION
        toPeriodFilename = args.toPeriod + MODEL_FILE_EXTENSION

        modelA = fasttext.load_model(os.path.join(MODELS_FOLDER, fromPeriodFilename))
        modelB = fasttext.load_model(os.path.join(MODELS_FOLDER, toPeriodFilename))

        cosineDistances = {}
        for word in modelA.words:
            if word in modelB.words:
                cosineDistances[word] = getCosineDistanceOfVectors(modelA[word], modelB[word])

        sortedCosineDistances = sorted(cosineDistances.items(), key=lambda x: x[1], reverse=True)

        print(sortedCosineDistances[:10])
    elif args.action == 'getCS':
        logger.info('Selected action: Get cosine similarity')

        fromPeriodFilename = args.fromPeriod + MODEL_FILE_EXTENSION
        toPeriodFilename = args.toPeriod + MODEL_FILE_EXTENSION

        modelA = fasttext.load_model(os.path.join(MODELS_FOLDER, fromPeriodFilename))
        modelB = fasttext.load_model(os.path.join(MODELS_FOLDER, toPeriodFilename))

        cosineSimilarities = {}
        for word in modelA.words:
            if word in modelB.words:
                cosineSimilarities[word] = getCosineSimilarityOfVectors(modelA[word], modelB[word])

        sortedCosineSimilarities = sorted(cosineSimilarities.items(), key=lambda x: x[1])

        print(sortedCosineSimilarities[:10])

########################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------
########################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--version', action='version', version='0.0.2')
subparsers = parser.add_subparsers()

parser_metadata = subparsers.add_parser('metadata')
parser_metadata.add_argument('--printStandard', action='store_true', help='print the standard metadata')
parser_metadata.add_argument('--printEnhanced', action='store_true', help='print the enhanced metadata')
parser_metadata.add_argument('--export', action='store_true', help='export the enhanced metadata')
parser_metadata.add_argument('--exportTextByPeriod', action='store_true', help='export the text by period')
parser_metadata.set_defaults(func=metadataParser)

parser_model = subparsers.add_parser('model')
parser_model.add_argument('--action', default='getNN', help='target action.')
parser_model.add_argument('--word', help='target word to get nearest neighbours for.')
parser_model.add_argument('--period', help='the target period to load the model from.')
parser_model.add_argument('--textsFolder', default='./output/texts', help='the target folder that contains the texts files.')
parser_model.add_argument('--fromPeriod', default='1800', help='the target starting period.')
parser_model.add_argument('--toPeriod', default='1900', help='the target ending period.')
parser_model.set_defaults(func=modelParser)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)
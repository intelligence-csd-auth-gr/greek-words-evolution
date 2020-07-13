import fasttext
import glob
import logging
import os
import pexpect
import warnings


warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FASTTEXT_PATH = os.path.join(os.path.curdir, 'fastText', 'fasttext')
MODELS_FOLDER = os.path.join(os.path.curdir, 'output', 'models')
MODEL_FILE_EXTENSION = '.model'
TEXT_FILE_EXTENSION = '.txt'


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


# note it seems that there is a built in method for that used like that: model.get_nearest_neighbors('άνδρας')
def getNeighboursForWord(word, modelFilename, neighborsCount):
    process = pexpect.spawn('%s nn %s %d' % (FASTTEXT_PATH, os.path.join(MODELS_FOLDER, modelFilename), neighborsCount))
    process.expect('Query word?')  # Flush the first prompt out.
    process.sendline(word)
    process.expect('Query word?')
    output = process.before

    return [word] + [line.strip().split()[0] for line in output.decode('utf8').strip().split('\n')[1:]]

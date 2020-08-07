import fasttext
import glob
import logging
import os
import pexpect
import warnings
import lib.file as file
import lib.vector as vector

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def createModel(textsFilename, modelsFolder, modelFileExention, textFileExtension):
    """

    @param textsFilename:
    @param modelsFolder:
    @param modelFileExention:
    @param textFileExtension:
    @return:
    @rtype: None
    """
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
    modelFilename = os.path.join(modelsFolder, os.path.basename(textsFilename).replace(textFileExtension,
                                                                                       modelFileExention))
    model.save_model(modelFilename)


def createModelsFromTextFiles(textFilesFolder, textFileExtension, modelsFolder, modelFileExention):
    """

    @param textFilesFolder:
    @param textFileExtension:
    @param modelsFolder:
    @param modelFileExention:
    @return:
    @rtype: None
    """
    filenames = glob.glob(os.path.join(textFilesFolder, '*' + textFileExtension))

    if (len(filenames) == 0):
        logger.info('No files detected in %s', textFilesFolder)
    else:
        for filename in filenames:
            if os.stat(filename).st_size != 0:
                with open(filename, 'r') as file:
                    contents = file.read().strip()
                    file.close()

                    if len(contents) != 0:
                        createModel(filename, modelsFolder, modelFileExention, textFileExtension)
                    else:
                        logger.info('File %s is empty. Skipping the model creation.', filename)


# note it seems that there is a built in method for that used like that: model.get_nearest_neighbors('άνδρας')
def getNeighboursForWord(word, modelFilename, modelsFolder, fasttextPath, neighborsCount):
    """

    @param word:
    @param modelFilename:
    @param modelsFolder:
    @param fasttextPath:
    @param neighborsCount:
    @return:
    @rtype: list
    """
    targetFile = os.path.join(modelsFolder, modelFilename)

    if not os.path.exists(targetFile):
        logger.info('File %s does not exist. Skipping..', modelFilename)
    else:
        process = pexpect.spawn('%s nn %s %d' % (fasttextPath, targetFile, neighborsCount))
        process.expect('Query word?')  # Flush the first prompt out.
        process.sendline(word)
        process.expect('Query word?')
        output = process.before

        return [word] + [line.strip().split()[0] for line in output.decode('utf8').strip().split('\n')[1:]]


def exportByDistance(action, modelFileExtension, modelsFolder, fromYear, toYear, neighborsCount, fasttextPath):
    """

    @param action:
    @param modelFileExtension:
    @param modelsFolder:
    @param fromYear:
    @param toYear:
    @param neighborsCount:
    @param fasttextPath:
    @return:
    @rtype: None
    """
    fromYearFilename = fromYear + modelFileExtension
    toYearFilename = toYear + modelFileExtension

    modelA = fasttext.load_model(os.path.join(modelsFolder, fromYearFilename))
    modelB = fasttext.load_model(os.path.join(modelsFolder, toYearFilename))

    clearVectorModelA = {}
    clearVectorModelB = {}

    for label in modelA.get_labels():
        clearVectorModelA[label] = modelA.get_word_vector(label)

    for label in modelB.get_labels():
        clearVectorModelB[label] = modelB.get_word_vector(label)

    # alignedEmbeddingsB = vector.alignTwoEmbeddings(clearVectorModelA, clearVectorModelB)

    results = {}
    for word in modelA.words:
        if word in modelB.words:
            if action == 'getCD':
                results[word] = vector.getCosineDistance(clearVectorModelA[word], clearVectorModelB[word])
            elif action == 'getCS':
                results[word] = vector.getCosineSimilarity(clearVectorModelA[word], clearVectorModelB[word])

    if action == 'getCD':
        sortedResults = sorted(results.items(), key=lambda x: x[1], reverse=True)
    elif action == 'getCS':
        sortedResults = sorted(results.items(), key=lambda x: x[1])

    resultsPerPeriod = {}

    for wordTuple in sortedResults[:50]:
        word = wordTuple[0]
        resultsPerPeriod[word] = {}

        resultsPerPeriod[word][str(fromYear)] = getNeighboursForWord(word, fromYearFilename, modelsFolder,
                                                                     fasttextPath, neighborsCount)
        resultsPerPeriod[word][str(toYear)] = getNeighboursForWord(word, toYearFilename, modelsFolder,
                                                                   fasttextPath, neighborsCount)

    # print(resultsPerPeriod)
    file.exportTextToFile(resultsPerPeriod, './shifts.json', True)

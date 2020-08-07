import codecs
import json
import logging
import os
import pandas
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def readMetadata(filename):
    """

    @param filename:
    @return: dataFrame
    @rtype: DataFrame
    """
    logger.info('Reading metadata from %s', filename)

    dataFrame = pandas.read_csv(filename, sep='\t')

    dataFrame['text'] = ''

    return dataFrame


def getContents(filename):
    """

    @param filename:
    @return:
    @rtype: list
    """
    if os.path.isfile(filename):
        with codecs.open(filename, 'r', 'utf-8') as file:
            fileContents = file.read().splitlines()
        file.close()
    else:
        fileContents = []

    return fileContents


def exportTextToFile(text, filename, exportToJson=False):
    """

    @param text:
    @param filename:
    @param exportToJson:
    @return:
    @return: None
    """
    fileHandler = open(filename, 'w', encoding='utf8')

    if exportToJson:
        json.dump(text, fileHandler, indent=4, ensure_ascii=False)
    else:
        fileHandler.write(text)

    fileHandler.close()

import pandas
import lib.file as file
import lib.text as text


def getCombined(corpora, targetCorpus, shouldEnhance=False):
    """

    @param corpora:
    @param targetCorpus:
    @param shouldEnhance:
    @return:
    @rtype: DataFrame
    """
    metadataList = []

    for corpus in corpora:
        if ((targetCorpus == 'all') or (targetCorpus and corpus['name'] == targetCorpus)):
            targetMetadata = file.readMetadata(corpus.get('metadataFilename'))

        if (shouldEnhance):
            textFilesFolder = corpus.get('textFilesFolder')
            enhancedMetadata = text.enhanceMetadata(textFilesFolder, metadata=targetMetadata, detectPublishedYear=False,
                                                    calculateTokens=False)

            metadataList.append(enhancedMetadata)
        else:
            metadataList.append(targetMetadata)

    return pandas.concat(metadataList)

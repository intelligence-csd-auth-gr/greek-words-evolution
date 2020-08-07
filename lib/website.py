import csv

import lib.websites.openbook as openbook


def fetchLinks(target):
    """
    @param target:
    @return:
    """
    # TODO validate the incoming target and use the appropriate website rather than hardcoding to "openbook"
    return openbook.getPostUrls()


def fetchMetadata(target, downloadFilenameExtension, metadataFilename):
    """
    @param target:
    @param downloadFilenameExtension:
    @param metadataFilename:
    @return:
    @rtype: DataFrame
    """
    postUrls = fetchLinks(target)

    postsMetadata = openbook.parsePosts(postUrls, downloadFilenameExtension)
    openbook.writeMetadataToCSV(postsMetadata, metadataFilename)

    return postsMetadata


def fetchFiles(target, downloadFilenameExtension, metadataFilename, downloadFolder):
    """
    @param target:
    @param downloadFilenameExtension:
    @param metadataFilename:
    @param downloadFolder:
    @return:
    @rtype: DataFrame
    """
    fetchMetadata(target, downloadFilenameExtension, metadataFilename)

    with open(metadataFilename) as fileHandler:
        postsMetadata = csv.DictReader(fileHandler, delimiter=',')
        openbook.downloadAttachments(postsMetadata, downloadFolder)

    return postsMetadata

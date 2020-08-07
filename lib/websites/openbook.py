from bs4 import BeautifulSoup
import csv
import os.path
import requests
import re
import urllib

########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################

BASE_URL = 'https://www.openbook.gr/category/literature/page/'
SOUP_METADATA_HTML_CONFIGURATION = {'class_': 'post-content description'}
SOUP_POST_ELEMENTS_CONFIGURATION = {'name': 'article', 'class_': 'post'}
SOUP_POST_ELEMENT_CONFIGURATION = {'name': 'a', 'class_': 'image-link'}
START_PAGE = 0
END_PAGE = 1

########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################


def clearText(text):
    """
    @param text:
    @return:
    @rtype: str
    """
    return re.sub('[»«‒§—·•]', '', text).strip()


def extractText(textContent, searchString):
    """
    @param textContent:
    @param searchString:
    @return:
    @rtype: str
    """
    matchingElements = re.search(r'(' + re.escape(searchString) + ':?)(.*)', textContent, re.MULTILINE | re.IGNORECASE)

    if (matchingElements):
        return clearText(matchingElements.group(2))

    return ''


def extractTitle(textContent):
    """
    @param textContent:
    @return:
    @rtype: str
    """
    return extractText(textContent, 'Τίτλος')


def extractAuthor(textContent):
    """
    @param textContent:
    @return:
    @rtype: str
    """
    return extractText(textContent, 'Συγγραφέας')


def extractType(textContent):
    """
    @param textContent:
    @return:
    @rtype: str
    """
    return extractText(textContent, 'Είδος')


def extractPublishedYear(textContent):
    """
    @param textContent:
    @return:
    @rtype: str
    """
    return extractText(textContent, 'Έτος έκδοσης')


def extractISBN(textContent):
    """

    @param textContent:
    @return:
    @rtype: str
    """
    return extractText(textContent, 'ISBN')


def extractAttachmentUrl(soupContent):
    """

    @param soupContent:
    @return:
    @rtype: str
    """
    htmlElement = soupContent.find(class_='wpcmsdev-button')

    if htmlElement:
        return htmlElement['href']

    return ''


def getPostUrls():
    """

    @return:
    @rtype: list
    """
    postUrls = []

    for pageNumber in range(START_PAGE, END_PAGE):
        print('Retrieving posts from page ' + str(pageNumber))

        pageUrl = BASE_URL + str(pageNumber) + '/'
        pageHtml = requests.get(pageUrl)
        soup = BeautifulSoup(pageHtml.text, 'html.parser')
        soupElements = soup.findAll(**SOUP_POST_ELEMENTS_CONFIGURATION)
        for soupElement in soupElements:
            postUrl = soupElement.find(**SOUP_POST_ELEMENT_CONFIGURATION)['href']
            postUrls.append(postUrl)

    return postUrls


def parsePosts(postUrls, downloadFileNameExtension):
    """

    @param postUrls:
    @param downloadFileNameExtension:
    @return:
    @rtype: list
    """
    postsMetadata = []

    for index, postUrl in enumerate(postUrls, start=1):
        print('Retrieving metadata for ' + postUrl)

        postHtml = requests.get(postUrl)
        soup = BeautifulSoup(postHtml.text, 'html.parser')
        soupElement = soup.find(**SOUP_METADATA_HTML_CONFIGURATION)
        textElement = soup.text

        id = 'openBook' + str(index)
        title = extractTitle(textElement)
        author = extractAuthor(textElement)
        type = extractType(textElement)
        publishedYear = extractPublishedYear(textElement)
        isbn = extractISBN(textElement)
        attachmentUrl = extractAttachmentUrl(soupElement)
        filename = id + downloadFileNameExtension

        postMetadata = {
            'id': id,
            'title': title,
            'author': author,
            'type': type,
            'publishedYear': publishedYear,
            'isbn': isbn,
            'filename': filename,
            'postUrl': postUrl,
            'attachmentUrl': attachmentUrl
        }

        postsMetadata.append(postMetadata)

    return postsMetadata


def writeMetadataToCSV(postsMetadata, metadataFilename):
    """

    @param postsMetadata:
    @param metadataFilename:
    @return:
    @rtype: None
    """
    csvColumns = ['id', 'title', 'author', 'type', 'publishedYear', 'isbn', 'filename', 'postUrl', 'attachmentUrl']

    try:
        with open(metadataFilename, 'w') as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=csvColumns)
            writer.writeheader()

            for postMetadata in postsMetadata:
                writer.writerow(postMetadata)
    except IOError:
        print('I/O error while writing to .csv file', IOError)


def downloadAttachments(postsMetadata, downloadFolder):
    """

    @param postsMetadata:
    @param downloadFolder:
    @return:
    @rtype: None
    """
    for postMetadata in postsMetadata:
        if postMetadata['attachmentUrl'] and not os.path.isfile(os.path.join(downloadFolder,
                                                                             postMetadata['filename'])):
            print('Downloading file "' + postMetadata['filename'] + '" from ' + postMetadata['attachmentUrl'])
            try:
                urllib.request.urlretrieve(postMetadata['attachmentUrl'], os.path.join(downloadFolder,
                                                                                       postMetadata['filename']))
            except urllib.error.HTTPError as e:
                print('Download error')
                print(e)

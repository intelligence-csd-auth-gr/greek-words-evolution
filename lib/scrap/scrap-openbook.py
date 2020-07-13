from bs4 import BeautifulSoup
import csv
import os.path
import pickle
import requests
import re
from tika import parser
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
DOWNLOAD_FOLDER = 'pdf'
DOWNLOAD_FILENAME_EXTENSION = '.pdf'
TEXT_FOLDER = 'text'
TEXT_FILENAME_EXTENSION = '.txt'
POST_URLS_FILENAME = 'post_urls.pickle'
METADATA_FILENAME = 'raw_metadata.csv'

########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################


def clearText(text):
    return re.sub('[»«‒§—·•]', '', text).strip()


def extractText(textContent, searchString):
    matchingElements = re.search(r'(' + re.escape(searchString) + ':?)(.*)', textContent, re.MULTILINE | re.IGNORECASE)

    if (matchingElements):
        return clearText(matchingElements.group(2))

    return ''


def extractTitle(textContent):
    return extractText(textContent, 'Τίτλος')


def extractAuthor(textContent):
    return extractText(textContent, 'Συγγραφέας')


def extractType(textContent):
    return extractText(textContent, 'Είδος')


def extractPublishedYear(textContent):
    return extractText(textContent, 'Έτος έκδοσης')


def extractISBN(textContent):
    return extractText(textContent, 'ISBN')


def extractAttachmentUrl(soupContent):
    htmlElement = soupContent.find(class_='wpcmsdev-button')

    if htmlElement:
        return htmlElement['href']

    return ''


def getPostUrls():
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


def parsePosts(postUrls):
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
        filename = id + DOWNLOAD_FILENAME_EXTENSION

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


def writeMetadataToCSV(postsMetadata):
    csvColumns = ['id', 'title', 'author', 'type', 'publishedYear', 'isbn', 'filename', 'postUrl', 'attachmentUrl']

    try:
        with open(METADATA_FILENAME, 'w') as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=csvColumns)
            writer.writeheader()

            for postMetadata in postsMetadata:
                writer.writerow(postMetadata)
    except IOError:
        print('I/O error while writing to .csv file', IOError)


def downloadAttachments(postsMetadata):
    for postMetadata in postsMetadata:
        if postMetadata['attachmentUrl'] and not os.path.isfile(os.path.join(DOWNLOAD_FOLDER,
                                                                             postMetadata['filename'])):
            print('Downloading file "' + postMetadata['filename'] + '" from ' + postMetadata['attachmentUrl'])
            try:
                urllib.request.urlretrieve(postMetadata['attachmentUrl'], os.path.join(DOWNLOAD_FOLDER,
                                                                                       postMetadata['filename']))
            except urllib.error.HTTPError as e:
                print('Download error')
                print(e)


def extractTextFromPdf(postsMetadata):
    for postMetadata in postsMetadata:
        pdfFilePath = os.path.join(DOWNLOAD_FOLDER, postMetadata['filename'])

        # check in case the metadata folder does not exist for some reason
        if (os.path.isfile(pdfFilePath)):
            textFileName = postMetadata['id'] + TEXT_FILENAME_EXTENSION
            textFilePath = os.path.join(TEXT_FOLDER, textFileName)

            # check in case we resume the conversion, in order to avoid reconverting the same files again
            if not (os.path.isfile(textFilePath)):
                parsed = parser.from_file(pdfFilePath)

                # check in case the parsed content is empty
                if (parsed and parsed.get('content')):
                    with open(textFilePath, 'w+') as textFileHandler:
                        textFileHandler.write(parsed['content'])


########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################

#####################################
# STEP 0 - Set up required folders and any perform any other preliminary tasks
#####################################
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

if not os.path.exists(TEXT_FOLDER):
    os.makedirs(TEXT_FOLDER)

########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
########################################################################################################################

#####################################
# STEP 1 - Retrieve all the post URLs to scrap
#####################################
###
# 1.1 - Retrieve them from scratch
###
postUrls = getPostUrls()

###
# 1.2 - Save them in a pickle file
###
with open(POST_URLS_FILENAME, 'wb') as fp:
    pickle.dump(postUrls, fp)

#####################################
# STEP 2 - Parse the posts and extract their metadata
#####################################
postsMetadata = parsePosts(postUrls)

#####################################
# STEP 3 - Save the extracted metadata of all posts into a CSV file
#####################################
writeMetadataToCSV(postsMetadata)

#####################################
# STEP 4 - Parse the saved CSV file containing the posts metadata and download each attachment
#####################################
with open(METADATA_FILENAME) as fileHandler:
    postsMetadata = csv.DictReader(fileHandler, delimiter=',')
    downloadAttachments(postsMetadata)

#####################################
# STEP 4 - Extract text from the the downloaded PDF files
#####################################
with open(METADATA_FILENAME) as fileHandler:
    postsMetadata = csv.DictReader(fileHandler, delimiter=',')
    extractTextFromPdf(postsMetadata)

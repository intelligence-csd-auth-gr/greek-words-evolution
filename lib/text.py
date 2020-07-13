import enchant
import re
import nltk
import random
import os
import glob
import shutil
import logging
import time
import warnings
import lib.file as file

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEXT_FILE_EXTENSION = '.txt'
PRODUCED_TEXTS_FOLDER = '../output/texts'


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


def detectMalformed():
    dictionary = enchant.Dict('el_GR')
    stopWords = set(nltk.corpus.stopwords.words('greek'))
    filenames = glob.glob('../data/corpora/openbook/text/malformed/*.txt')

    malformedFiles = []

    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as file:
            clean = re.sub(r'\s\s+', ' ', file.read())
            clean = re.compile(r'\b(' + r'|'.join(stopWords) + r')\b\s*').sub('', clean)
            clean = re.sub(r'[^Α-Ωα-ωίϊΐόάέύϋΰήώ\s]', '', clean)
            clean = re.sub('[ἀἁἂἃἄἅἆἇὰάᾀᾁᾂᾃᾄᾅᾆᾇᾰᾱᾲᾳᾴᾶᾷ]', 'α', clean)
            clean = re.sub('[ἈἉἊἋἌἍἎἏᾈᾉᾊᾋᾌᾍᾎᾏᾸᾹᾺΆᾼ]', 'Α', clean)
            clean = re.sub('[ἐἑἒἓἔἕὲέ]', 'ε', clean)
            clean = re.sub('[ἙἚἛἜἝ]', 'Ε', clean)
            clean = re.sub('[ἠἡἢἣἤἥἦἧῆὴῇ]', 'η', clean)
            clean = re.sub('[ἨἩἪἫἬἭἮἯ]', 'Η', clean)
            clean = re.sub('[ἰἱἲἳἴἵὶῖ]', 'ι', clean)
            clean = re.sub('[ἶἷἸἹἺἻἼἽἾἿ]', 'Ι', clean)
            clean = re.sub('[ὀὁὂὃὄὅὸ]', 'ο', clean)
            clean = re.sub('[ὈὉὊὋὌὍ]', 'Ο', clean)
            clean = re.sub('[ὐὑὒὓὔὕὖὗ]', 'υ', clean)
            clean = re.sub('[ὙὛὝὟ]', 'Υ', clean)
            clean = re.sub('[ὠὡὢὣὤὥὦὧῶ]', 'ω', clean)
            clean = re.sub('[ὨὩὪὫὬὭὮὯ]', 'Ω', clean)

            tokens = nltk.word_tokenize(clean)
            if len(tokens) > 100:
                sampleSize = 100
                sample = random.choices(tokens, k=sampleSize)
                # print(sample)
                results = [dictionary.check(item) for item in sample]
                if sum(results) < 85:
                    print(sample)
                    print(filename)
                    malformedFiles.append(filename)
                    # shutil.move(filename, './data/corpora/openbook/text/malformed/' + os.path.basename(filename))
                else:
                    shutil.move(filename, '../data/corpora/openbook/text/parsable/' + os.path.basename(filename))
            else:
                shutil.move(filename, '../data/corpora/openbook/text/no-contents/' + os.path.basename(filename))


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
        textFileContentsAsLines = file.getContents(os.path.join(textFilesFolder, row['id'] + TEXT_FILE_EXTENSION))
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
        file.exportTextToFile(preProcessedText, os.path.join(PRODUCED_TEXTS_FOLDER, ('%s' + TEXT_FILE_EXTENSION) % i))

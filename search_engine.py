import os
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def readFiles():
    """Reads files provided in corpus.

    Returns:
        Normalized dict of TF-IDF vector for each document. The second parameter
        is total no. files read.
    """
    corpusroot = './presidential_debates'
    documents = {}
    documentCount = 0
    for filename in os.listdir(corpusroot):
        file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
        doc = file.read()
        file.close()
        doc = doc.lower()
        # print(file)
        tokens = tokenize(doc)
        tokensWithoutStopWords = removeStopWords(tokens)
        stemmedTokens = applyStemmer(tokensWithoutStopWords)
        documents[filename] = createDocumentTFVector(stemmedTokens)
        documentCount += 1
        # break
    documentDFVector = createDocumentDFVector(documents)
    doumentTF_IDFVector = createDoumentTF_IDFVector(documentDFVector, documentCount)
    normalizeDoumentTF_IDFVector = createNormalizeDoumentTF_IDFVector(doumentTF_IDFVector)
    # print(normalizeDoumentTF_IDFVector)
    return normalizeDoumentTF_IDFVector, documentCount

def tokenize(doc):
    """Tokenize the document provided as input.

    Args:
        doc: content of the file

    Returns:
        List of tokens

    """
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(doc)
    # print(tokens)
    return tokens

def removeStopWords(tokens):
    """Remove the English stop words from the tokens list.

    Args:
        tokens: Accepst list of tokens

    Returns:
        List of tokens without English stop words

    """
    # print(stopwords.words('english'))
    # print(sorted(stopwords.words('english')))
    stopWords = stopwords.words('english')
    tokensWithoutStopWords = []
    for token in tokens:
        if token not in stopWords:
            tokensWithoutStopWords.append(token)
    # print(len(tokens))
    # print(len(tokensWithoutStopWords))
    return tokensWithoutStopWords

def applyStemmer(tokensWithoutStopWords):
    """Apply porter stemmer on List of tokens.

    Args:
        tokensWithoutStopWords: Accepts List of tokens

    Returns:
        List of tokens with porter stemmer applied
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokensWithoutStopWords]

def createDocumentTFVector(stemmedTokens):
    """Create TF vector from stemmed List of tokens.

    Args:
        stemmedTokens: List of stemmed tokens
    Returns:
        A dict of token and its frequency
    """
    tokenDict = {}
    for token in stemmedTokens:
        if token in tokenDict:
            tokenDict[token]['tf'] +=1
        else:
            tokenDict[token]= {'tf':1}
    # print(tokenDict)
    return tokenDict

def createDocumentDFVector(documents):
    """Calculate DF for each token.

    Args:
        documents: A dict of documents, where each document is again dict of tokens

    Returns:
        A dict of document where for each token dict contains additional field
        called df which represents document frequency for each token

    """
    for document in documents:
        for token in documents[document]:
            count = 0
            for documentForTokenCheck in documents:
                if token in documents[documentForTokenCheck]:
                    count +=1
            documents[document][token]['df'] = count
    return documents

def createDoumentTF_IDFVector(documents, documentCount):
    """Calculate TF-IDF vector for each document.

    Args:
        documents: A dict of documents where each document represents
        dict of tokens it contains
        documentCount: Total number of documents
    Returns:
        A dict of document where for each token dict contains additional field
        called Wtd which represents TF-IDF vector
    """

    for document in documents:
        for token in documents[document]:
            documents[document][token]['Wtd'] = (1 + math.log10(documents[document][token]['tf'])) *(math.log10(documentCount/documents[document][token]['df']))
    return documents

def createNormalizeDoumentTF_IDFVector(documents):
    """Normalize TF-IDF vector.

    Args:
        documents:A dict of documents where each document represents dict
        of tokens it contains

    Returns:
        Returns:
            A dict of document where for each token dict contains additional
            field called NWtd which represents Normalized TF-IDF vector

    """
    for document in documents:
        squareSummation = 0
        for token in documents[document]:
            squareSummation += (documents[document][token]['Wtd'])*(documents[document][token]['Wtd'])

        squareRootSummation = math.sqrt(squareSummation)
        sum = 0
        for token in documents[document]:
            documents[document][token]['NWtd'] = (documents[document][token]['Wtd'])/squareRootSummation
            sum += documents[document][token]['NWtd']*documents[document][token]['NWtd']
        # print('---------------------------------------', math.sqrt(sum))
    return documents

def processQueryString(query):
    """Processing input query string to be matched.

    Args:
        query: A string which is used to find matching document

    Returns:
        A dict of tokens from query string, which has normalized TF-IDF value

    """
    query = query.lower()
    tokens =  tokenize(query)
    tokensWithoutStopWords = removeStopWords(tokens)
    stemmedTokens = applyStemmer(tokensWithoutStopWords)
    queryTFVector = createDocumentTFVector(stemmedTokens)
    queryWTFVector = createWeightedTFVector(queryTFVector)
    queryNWTFVector = createNormalizeQueryWTFVector(queryWTFVector)
    # print(queryNWTFVector)
    return queryNWTFVector

def createWeightedTFVector(queryTFVector):
    """Create weighted qwery vector for given TF vector.

    Args:
        queryTFVector: A dict of tokens from query string with tf field

    Returns:
        A dict of tokens with extra field wtd, which represents weighted tf
        value

    """
    for token in queryTFVector:
        queryTFVector[token]['Wtd']= (1 + math.log10(queryTFVector[token]['tf']))
    return queryTFVector

def createNormalizeQueryWTFVector(queryWTFVector):
    """Create normalized weighted qwery vector for given TF vector.

    Args:
        queryWTFVector: A dict of tokens from query string with wtf field

    Returns:
        A dict of tokens with extra field Nwtd, which represents weighted tf
        value

    """
    squareSummation = 0
    for token in queryWTFVector:
        squareSummation += (queryWTFVector[token]['Wtd'])*(queryWTFVector[token]['Wtd'])
    squareRootSummation = math.sqrt(squareSummation)
    sum = 0
    for token in queryWTFVector:
        queryWTFVector[token]['NWtd']= queryWTFVector[token]['Wtd']/squareRootSummation
        sum += queryWTFVector[token]['NWtd']*queryWTFVector[token]['NWtd']
    # print('---------------------------------------', math.sqrt(sum))
    return queryWTFVector

def cosineMatch(documents, queryWTFVector):
    """Naive implementation of cosine Similiarity.

    Args:
        documents: A dict of documents, where each document is again dict of tokens
        queryWTFVector: A normalized query vector

    """
    resultDocument = None
    initialDocumentCosineScore = 0
    for document in documents:
        dotProduct = 0
        print('-----------------------------')
        for token in queryWTFVector:
            if token in documents[document]:
                print(token, documents[document][token])
                dotProduct += queryWTFVector[token]['NWtd']*documents[document][token]['NWtd']
        print(document, dotProduct)
        if dotProduct > initialDocumentCosineScore:
            initialDocumentCosineScore = dotProduct
            resultDocument = document
        # print('-----------------------------')

    print('finanl result------------',resultDocument, initialDocumentCosineScore)

def buildPostingList(documents):
    """Buildig postingList data structure from given dict of documents.

    Args:
        documents: A dict of documents, where each document is again dict of tokens

    Returns:
        A dictionary of token as key and List as value, which represents
        documents in descending order of normalized TF-IDF value

    """
    postingList = {}
    for document in documents:
        for token in documents[document]:
            if token not in postingList:
                postingList[token]=[]
                postingList[token].append({'document' : document, 'tokenData' : documents[document][token]})
            else:
                postingList[token].append({'document' : document, 'tokenData' : documents[document][token]})
    # print('----------------postingList ---------------------')
    # print(postingList)

    for listItem in postingList:
        tempList = sorted(postingList[listItem], key = lambda data:data['tokenData']['NWtd'], reverse=True)
        postingList[listItem] = tempList

    # print('----------------postingList sorted---------------------')
    # print(postingList)
    return postingList

def matchWithPostingList(postingList, queryNWTFVector):
    """Matching query to the documents based PostingList Data structure.

    Args:
        postingList: A dictionary of token as key and List as value, which represents
        documents in descending order of normalized TF-IDF value
        queryNWTFVector: normalized query vector

    Returns:
        if highest matching document is found then it returns document name
        and cosine match score. If none of the document contains the tokens then
        it returns None and cosine score 0. If actual score of the document is
        less than upper bound score, then it returns fetch more and 0

    """

    documentList = {}
    for term in queryNWTFVector:
        index = 0
        # documentList[term] = []
        if term in postingList:
            documentList[term] = []
            for item in postingList[term]:
                documentList[term].append(item)
                index += 1
                if index == 10:
                    break

    actualScoreDict = {}
    upperLimitScoreDict= {}
    for token in documentList:
        # print(token,'++++++++++++++++++++++++++++++++++')
        for document in documentList[token]:
            # print(document, '\n')
            if document['document'] in actualScoreDict or document['document'] in upperLimitScoreDict:
                pass
            else:
                score = 0.0
                isActualScore = True
                for token1 in documentList:
                    documentFound = False
                    for documentItem in documentList[token1]:
                        if document['document'] in documentItem['document']:
                            score +=  documentItem['tokenData']['NWtd'] * queryNWTFVector[token]['NWtd']
                            documentFound = True
                            break
                    if not documentFound:
                        isActualScore = False
                        if len(documentList[token1]) == 10:
                            score += documentList[token1][9]['tokenData']['NWtd']*queryNWTFVector[token]['NWtd']
                        else:
                            score += 0.0 *queryNWTFVector[token1]['NWtd']

                if isActualScore:
                    actualScoreDict[document['document']]= score
                else:
                    upperLimitScoreDict[document['document']]= score

    # print("actual score ----", actualScoreDict)
    # print("upper score -----", upperLimitScoreDict)
    if len(actualScoreDict) == 0 and len(upperLimitScoreDict) == 0:
        return None, 0.0
    else:
        maxActualScore = 0.0
        maxActualDocument = None
        for doc in actualScoreDict:
            if actualScoreDict[doc] > maxActualScore:
                maxActualScore = actualScoreDict[doc]
                maxActualDocument = doc

        upperLimitScore= 0.0
        upperLimitDocument = None
        for doc in upperLimitScoreDict:
            if upperLimitScoreDict[doc] > upperLimitScore:
                upperLimitScore = upperLimitScoreDict[doc]
                upperLimitDocument = doc
        if maxActualScore >= upperLimitScore:
            return maxActualDocument, maxActualScore
        else:
            return "fetch More", 0.0

#------------------------- preparing section-------------------------------
# global variables
documents, documentCount = readFiles()
postingList = buildPostingList(documents)

#------------------------ Interface section -------------------------------

def query(qstring):
    """Accepts query to match the most relevant document.

    Args:
        query:  A string which is used to find matching document

    Returns:
        matching highest matching document and its cosine Similiarity score

    """
    queryWTFVector = processQueryString(qstring)
    return matchWithPostingList(postingList, queryWTFVector)

def getweight(filename, token):
    """Get normalized TF_IDF weight of token form given file.

    Args:
        filename: name of the file from which token's TF-IDF score is taken
        token : string representing token
    Returns:
        normalized TF-IDF scor of the token
    """
    if filename in documents:
        if token in documents[filename]:
            return documents[filename][token]['NWtd']
        else:
            return 0
    else:

        return 0

def getidf(token):
    """Get idf score the token .

    Args:
        token : string representing token

    Returns:
        idf score of the token
    """
    for document in documents:
        if token in documents[document]:
            return math.log10(documentCount/documents[document][token]['df'])
    return -1

# --------------------------------- Query  Testing Section --------------------

# print("(%s, %.12f)" % query("health insurance wall street"))
# print("(%s, %.12f)" % query("particular constitutional amendment"))
# print("(%s, %.12f)" % query("terror attack"))
# print("(%s, %.12f)" % query("vector entropy"))
# print("%.12f" % getweight("2012-10-03.txt","health"))
# print("%.12f" % getweight("1960-10-21.txt","reason"))
# print("%.12f" % getweight("1976-10-22.txt","agenda"))
# print("%.12f" % getweight("2012-10-16.txt","hispan"))
# print("%.12f" % getweight("2012-10-16.txt","hispanic"))
# print("%.12f" % getidf("health"))
# print("%.12f" % getidf("agenda"))
# print("%.12f" % getidf("vector"))
# print("%.12f" % getidf("reason"))
# print("%.12f" % getidf("hispan"))
# print("%.12f" % getidf("hispanic"))
#
# print('rogue input')
#
# print("(%s, %.12f)" % query("       health insurance wall         street"))
# print("(%s, %.12f)" % query("particular             constitutional amendment "))
# print("(%s, %.12f)" % query(" Terror Attack ANiket "))
# print("(%s, %.12f)" % query("     vector entropy "))
# print("(%s, %.12f)" % query("       "))
# print("%.12f" % getweight("   ","agenda"))
# print("%.12f" % getweight("1976-10-22 ","agenda"))
# print("%.12f" % getweight("1976-10-22.txt","     "))
# print("%.12f" % getweight(" ","uoiuo"))
# print("%.12f" % getweight(" ","     "))
# print("%.12f" % getidf("     "))
# -----------------------------------------------------------------------------

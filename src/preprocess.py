import numpy as np
import cPickle as pkl
from nltk import FreqDist
import gzip,json

configuration = json.load(open('configurations/configuration_preprocess_article.json'))[0]

minDistance   = -30
maxDistance   = 30

words = {}
maxSentenceLen     = [0,0,0]
labelsDistribution = FreqDist()

labelsMapping      = {'None':0, 'rd-disab':1}
distanceMapping    = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}

for dis in xrange(minDistance+1,maxDistance+1):
    distanceMapping[dis] = len(distanceMapping)

files = [configuration["folder"]+'train-todo.txt']

def createMatrices(files, word2Idx, maxSentenceLen=100):
    labels          = []
    positionMatrix1 = []
    positionMatrix2 = []
    tokenMatrix     = []

    for line in open(files):
        splits   = line.strip().split('\t')
        label    = splits[0]
        pos1     = splits[1]
        pos2     = splits[2]
        sentence = splits[3]
        tokens   = sentence.split(" ")

        labelsDistribution[label] += 1

        tokenIds        = np.zeros(maxSentenceLen)
        positionValues1 = np.zeros(maxSentenceLen)
        positionValues2 = np.zeros(maxSentenceLen)

        for idx in xrange(0, min(maxSentenceLen, len(tokens))):
            tokenIds[idx] = getWordIdx(tokens[idx], word2Idx)

            distance1 = idx - int(pos1)
            distance2 = idx - int(pos2)

            if distance1 in distanceMapping:
                positionValues1[idx] = distanceMapping[distance1]
            elif distance1 <= minDistance:
                positionValues1[idx] = distanceMapping['LowerMin']
            else:
                positionValues1[idx] = distanceMapping['GreaterMax']

            if distance2 in distanceMapping:
                positionValues2[idx] = distanceMapping[distance2]
            elif distance2 <= minDistance:
                positionValues2[idx] = distanceMapping['LowerMin']
            else:
                positionValues2[idx] = distanceMapping['GreaterMax']

        tokenMatrix.append(tokenIds)
        positionMatrix1.append(positionValues1)        
        positionMatrix2.append(positionValues2)
        labels.append(labelsMapping[label])
    return np.array(labels, dtype='int32'), np.array(tokenMatrix, dtype='int32'), np.array(positionMatrix1, dtype='int32'), np.array(positionMatrix2, dtype='int32'),




def getWordIdx(token, word2Idx):
    """Returns from the word2Idex table the word index for a given token"""
    if token in word2Idx:
        return word2Idx[token]
    if token.lower() in word2Idx:
        return word2Idx[token.lower()]
    return word2Idx["UNKNOWN"]


for fileIdx in xrange(len(files)):
    for line in open(files[fileIdx],"rb").read().decode("utf-8").strip().split("\n"):
        splits   = line.strip().split('\t')
        tokens   = splits[3].split(" ")
        maxSentenceLen[fileIdx]  = max(maxSentenceLen[fileIdx], len(tokens))
        for token in tokens:
            words[token.lower()] = True

print("Max Sentence Lengths: ",maxSentenceLen)

word2Idx   = {}
embeddings = [] 

fEmbeddingsvoc = gzip.open(configuration["embeddingsPathvoc"]) if configuration["embeddingsPathvoc"].endswith('.gz') else open(configuration["embeddingsPathvoc"])
fEmbeddingsvec = gzip.open(configuration["embeddingsPathvec"]) if configuration["embeddingsPathvec"].endswith('.gz') else open(configuration["embeddingsPathvec"])


for line in fEmbeddingsvec:
    split_voc = fEmbeddingsvoc.next().strip().split("\t")
    split = line.strip().split("\t")

    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING"] = len(word2Idx)
        embeddings.append(np.zeros(len(split)))

        word2Idx["UNKNOWN"] = len(word2Idx)
        embeddings.append(np.random.uniform(-0.25, 0.25, len(split)))

    if split[0].lower() in words:
        embeddings.append(np.array([float(num) for num in split[0:]]))
        word2Idx[split_voc[0]] = len(word2Idx)



embeddings = np.array(embeddings)

print("Embeddings shape: ", embeddings.shape)
print("Len words: "       , len(words))

f = gzip.open(configuration["embeddingsPklPath"], 'wb')
pkl.dump(embeddings, f, -1)
f.close()

# :: Create token matrix ::
train_set = createMatrices(files[0], word2Idx, max(maxSentenceLen))


print("train_set = ", train_set)

# write sem-relations.pkl.gz 
f = gzip.open(configuration["outputFilePath"], 'wb')
pkl.dump(train_set, f, -1)
f.close()


print("Data stored in pkl folder")

for label, freq in labelsDistribution.most_common(100):
    print("%s : %f%%" % (label, 100*freq / float(labelsDistribution.N())))

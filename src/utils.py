def posToValue(idx,pos1, distanceMapping,minDistance):
    distance1 = idx - int(pos1)
    if distance1 in distanceMapping:
        return distanceMapping[distance1]
    elif distance1 <= minDistance:
        return distanceMapping['LowerMin']
    else:
        return distanceMapping['GreaterMax']

def getWordIdx(token, word2Idx):
    """Returns from the word2Idx table the word index for a given token"""
    if token in word2Idx:
        return word2Idx[token]
    if token.lower() in word2Idx:
        return word2Idx[token.lower()]
    return word2Idx["UNKNOWN"]

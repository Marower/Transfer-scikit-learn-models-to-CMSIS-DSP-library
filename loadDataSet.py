import csv

def extractFloatsFromRow(ToConverString):
    return [float(x) for x in ToConverString.split(',')]

def importFeatures():
    features = []
    with open('Data/features.csv', newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in csvreader:
            features.append(extractFloatsFromRow(row[0]))
    return features


def importLabels():
    labels = []
    with open('Data/labels.csv', newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in csvreader:
            labels.append(row[0])
    return labels

def getDataset():
    X = importFeatures()
    y = importLabels()

    X_train = X[0:1500]
    X_test = X[1500:len(X)]
    y_train = y[0:1500]
    y_test = y[1500:len(y)]
    return X_train, X_test, y_train, y_test
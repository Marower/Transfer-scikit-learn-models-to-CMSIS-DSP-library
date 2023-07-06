from loadDataSet import importLabels


def getClass (classLine):
    clas = classLine.split(': ')[-1].replace('\n', '')
    return clas


labels = importLabels()
start = 1500
fileLog = 'serial-20230126-091049.log'
fileRes = 'STMResults.csv'
flog = open(fileLog, 'r')
fres = open(fileRes, 'w')
fres.write("Correct label, True class, Naive Bayes, SVM_Linear, SVM_polynomial, SVM_rfb, SVM_Sigmoid\n")
for i in range(0, len(labels)-start):
    Line = flog.readline()
    number = Line.split(' ')[0]
    if number != str(i):
        print('error')
        break

    bayesLine = flog.readline()
    bayesClass = getClass(bayesLine)

    linearLine = flog.readline()
    linearClass = getClass(linearLine)

    polyLine = flog.readline()
    polyClass = getClass(polyLine)

    rbfLine = flog.readline()
    rbfClass = getClass(rbfLine)

    sigmoidLine = flog.readline()
    sigmoidClass = getClass(sigmoidLine)

    resultLine = ''
    if labels[start+i] == 'AF':
        resultLine = 'AF, 0, '
    else:
        resultLine = 'Normal, 1, '
    resultLine += bayesClass + ', ' + linearClass + ', ' + polyClass + ', ' + rbfClass + ', ' + sigmoidClass + '\n'
    fres.write(resultLine)

flog.close()
fres.close()

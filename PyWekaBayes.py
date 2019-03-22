import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.core.classes import JavaObject
import numpy as np
import weka.core.typeconv as typeconv
import javabridge
import weka.core.serialization as Serial

# from: http://pythonhosted.org/python-weka-wrapper/examples.html
# https://github.com/LeeKamentsky/python-javabridge/blob/master/javabridge/noseplugin.py
# https://gist.github.com/carl0967/1138b5a55d31ca1dda5c
# https://groups.google.com/forum/?utm_medium=email&utm_source=footer#!msg/python-weka-wrapper/qF4vw_6sqAA/EmqTph1NAAAJ

jvm.start(max_heap_size="4000M")

data_dir = "D:/weka/"

'''
5 Nodes for Iris dataset:

4 attributes:
node id: 0 (sepallength)    [-inf-5.55, 5.55-6.15, 6.15-inf]
node id: 1 (sepalwidth)     [-inf-2.95, 2.95-3.35, 3.35-inf]
node id: 2 (petallength)    [-inf-2.45, 2.45-4.75, 4.75-inf]
node id: 3 (petalwidth)     [-inf-0.8, 0.8-1.75, 1.75-inf]

1 class:
node id: 4 (class)          [0=iris-setosa, 1=iris-versicolor, 2=iris-virginica]
'''

# Choose trained Weka BIFXML file
xmlfile = "D:/weka/iris.xml"    # created with BayesNet, MaxNrParents=2, BIFXML file
bifreader = JavaObject(JavaObject.new_instance("weka.classifiers.bayes.net.BIFReader"))
editable = Classifier(jobject=javabridge.make_instance(
                "weka/classifiers/bayes/net/EditableBayesNet",
                "(Lweka/classifiers/bayes/net/BIFReader;)V",
                bifreader.jwrapper.processFile(xmlfile)))

# We need to calculate the margins of all the attributes
marginCalc = JavaObject(JavaObject.new_instance("weka.classifiers.bayes.net.MarginCalculator"))
marginCalc.jwrapper.calcMargins(editable.jobject)

marginCalcNoEvidence = Serial.deepcopy(marginCalc)    # could maybe get by without this, just use marginCalc()

# Have a look before we set evidence
print('Pre-evidence:\n', marginCalcNoEvidence)

# determine your class ID by name
classID = editable.jwrapper.getNode2('class')
print('class id:', classID)

for j in range(0,3):

    marginCalc = JavaObject(JavaObject.new_instance("weka.classifiers.bayes.net.MarginCalculator"))
    marginCalc.jwrapper.calcMargins(editable.jobject)

    # Set evidence to an index value of the class node to see which attributes are most influential
    marginCalc.jwrapper.setEvidence(classID, j)     #0=setosa, 1=versicolor, 2=virginica
    print('\n\n', editable.jwrapper.getNodeValue(classID, j))

    # View margins post-evidence
    print('Post-evidence:\n', marginCalc)

    # Get the margin on the attribute nodes(0-4)
    for i in range(0,4):
        gm = marginCalc.jwrapper.getMargin(i)              # I believe this could be 0 to 4
        arr = javabridge.get_env().get_double_array_elements(gm.o)
        print('Margin node', i, '(' + editable.jwrapper.getNodeName(i) + '):', arr)

jvm.stop()


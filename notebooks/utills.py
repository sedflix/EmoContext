import io
import re

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import json
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_feature_encoding
from varibles import *
import pickle
from keras.utils import to_categorical


def get_class_weight(y):
    """

    Used from: https://stackoverflow.com/a/50695814
    TODO: check validity and 'balanced' option
    :param y: A list of one-hot-encoding labels [[0,0,1,0],[0,0,0,1],..]
    :return: class-weights to be used by keras model.fit(.. class_weight="") -> {0:0.52134, 1:1.adas..}
    """
    y_integers = np.argmax(y, axis=1)
    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    d_class_weights = dict(enumerate(class_weights))
    return d_class_weights


def read_saved_features_vectors_file(file_x, file_y):
    """
    For sake of fast testing, I've vectorised all the text in the train.txt file. It is stored in the following format:
    file_x:
    a list of lists. Each list contains three arrays, each representing a vector for sentence_i
    (<number_of_samples>,<3, which is number of speakers>, <2304, feature vector len>)

    file_y:
    a list of strings -> ['others', 'angry', ...]

    both of them have the same length

    :param file_x:
    :param file_y:
    :return: x,y
    x -> array of arrays. Each array is of size 2304*3
    y -> labels corresponding to each index in x
    """
    print ("Starting to load features vectors from %s and %s" % (file_x, file_y))
    with open(file_x, 'rb') as f:
        x = pickle.load(f)
    with open(file_y, 'rb') as f:
        y = pickle.load(f)

    # TODO: Save data such that you don't have to change the it to categorical and concatenate
    for i in range(len(y)):
        y[i] = emotion2label[y[i]]
        x[i] = np.concatenate(x[i], axis=None)
    y = to_categorical(y)

    return np.array(x), y


def use_deepmoji(maxlen=MAXLEN,
                 vocab_path=DEEPMOJI_VOCAB_FILE,
                 weights_path=DEEPMOJI_WEIGHT_FILE):
    print('Tokenizing using dictionary from {}'.format(vocab_path))
    with open(vocab_path, 'r') as f:
        vocabulary = json.load(f)

    st = SentenceTokenizer(vocabulary, maxlen)

    print('Loading model from {}.'.format(weights_path))
    model = deepmoji_feature_encoding(maxlen, weights_path)
    model.summary()

    return st, model


def test_file_to_feature_vectors(test_file_path=TEST_DATA_FILE, is_label=True):
    """
    Read the file as given in the dataset
    :param test_file_path:
    :return:
    """
    df = pd.read_csv(test_file_path, sep='\t', header=(0), encoding='utf8')
    df.set_index('id')

    print ("Number of samples: %d", len(df))

    x = []
    y = []

    st, model = use_deepmoji()

    print ("Starting to convert text data to features")
    for i in range(len(df)):
        tokenized, _, _ = st.tokenize_sentences([df['turn1'][i], df['turn2'][i], df['turn3'][i]])
        encoding = model.predict(tokenized)
        x.append(encoding)
        if is_label:
            y.append(df['label'][i])
        if i % 1000 == 0:
            print ("Done %dth sample" % i)
    print ("Conversion Done")

    # #TODO: Save data such that you don't have to change the it to categorical and concatenate
    for i in range(len(x)):
        if is_label:
            y[i] = emotion2label[y[i]]
        x[i] = np.concatenate(x[i], axis=None)

    if is_label:
        y = to_categorical(y)
        return x, y
    else:
        return x


def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in  "train" mode] List of labels
    """
    indices = []
    conversations = []
    labels = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '
                line = cSpace.join(lineSplit)

            line = line.strip().split('\t')
            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                labels.append(label)

            conv = ' <eos> '.join(line[1:4])

            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)

            indices.append(int(line[0]))
            conversations.append(conv.lower())

    if mode == "train":
        return indices, conversations, labels
    else:
        return indices, conversations



def getMetrics(predictions, ground):
    """
    FROM: Baseline/starting_kit

    Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1))

    truePositives = np.sum(discretePredictions * ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground - discretePredictions, 0, 1), axis=0)

    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)

    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, 4):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = (2 * recall * precision) / (precision + recall) if (precision + recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))

    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision) / (macroPrecision + macroRecall) if (
                                                                                             macroPrecision + macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (
        macroPrecision, macroRecall, macroF1))

    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()

    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (
        truePositives, falsePositives, falseNegatives))

    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)

    microF1 = (2 * microRecall * microPrecision) / (microPrecision + microRecall) if (
                                                                                             microPrecision + microRecall) > 0 else 0
    # -----------------------------------------------------

    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions == ground)

    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (
        accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1

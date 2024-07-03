import evaluate, os
import pandas as pd
metric = evaluate.load("seqeval")

def compute_metrics(p, l):
    true_predictions = p
    true_labels = l
    results = metric.compute(predictions=true_predictions, references=true_labels)
    precision = results["overall_precision"]
    recall = results["overall_recall"]
    f1 = results["overall_f1"]
    accuracy = results["overall_accuracy"]
    return precision,recall,f1,accuracy

def data_process(data):

    accuracy = 0
    count = 0
    for index, row in data.iterrows():
        label = row['label'].strip().lower()
        label = label.split(',')[-1]
        label = label[:-1]
        label = label.replace("\"", "")
        label = label.replace("-", " ")

        response = str(row['response']).strip().lower()
        response = response.split("[/inst]")[-1]
        response = response.replace("\'", "")
        response = response.replace("-", " ")
        response = response.replace("\"", "")

        labels = label.split()
        responses = response.split()

        dic = {}
        for i in range(len(labels)):
            dic[i] = 0
        found = 0
        for r in responses:
            for index, l in enumerate(labels):
                if r == l:
                    dic[index] = 1
                    break
        sums = 0
        for i in dic:
            if dic[i] == 1:
                sums += 1
        if sums == len(labels):
            found = 1
            accuracy += 1
        if found == 0:
            if response in label or label in response:
                accuracy += 1
        count += 1
    return accuracy/count, count

directory='../prompt_and_response/entity_linking/'

files = os.listdir(directory)

for file in files:
    if '.csv' not in file:
        continue
    data = pd.read_csv(directory+file)
    accuracy, count = data_process(data)
    print(file, accuracy*100, count)

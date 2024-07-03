import evaluate, json, os
import pandas as pd
import copy
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

def data_process(filename, data):
    tags_to_labels_dic = {'o': 'O',
                          'b': 'B-MISC',
                          'i': 'I-MISC'}
    predicted_labels = []
    gold_labels = []
    for index, row in data.iterrows():
        p_l = []
        t_l = []
        labels = row['label'].lower().split("\n")
        response = []

        row['response'] = str(row['response'])
        if str(row['response']) == "nan":
            for label in labels:
                response.append('O')
        else:
            if "llama2" in filename:
                row['response'] = row['response'].split("[/INST]")[-1].lower()
                if "\\n\\n" in row['response']:
                    row['response'] = row['response'].split("\\n\\n")[1]
                    if "\\n\\n" in row['response']:
                        row['response'] = row['response'].split("\\n\\n")[0]
                row['response'] = row['response'].replace(".\"]", "")
                row['response'] = row['response'].replace("\"]", "")
                row['response'] = row['response'].replace("{", "")
                row['response'] = row['response'].replace("}", "")
                response = row['response'].strip().split("\\n")
            else:
                response = row['response'].strip().split("\n")

        predicts = response

        for pred in predicts:
            value = pred.split(':')[-1].strip()
            value = value.split('-')[0] # sometimes LLMs predict B-Gene, I-Gene, etc. So, we extract the first part.
            if value.lower() in tags_to_labels_dic:
                p_l.append(tags_to_labels_dic[value.lower()])
            else:
                p_l.append('O')

        for label in labels:
            label = label.lower().strip()
            if label not in tags_to_labels_dic:
                label = 'o'
            label = tags_to_labels_dic[label]
            t_l.append(label)

        if len(labels) > len(p_l):
            required_length = len(labels)
            modified_length = len(p_l)
            while modified_length!=required_length:
                p_l.append('O')
                modified_length = len(p_l)
        elif len(p_l) > len(labels):
            required_length = len(labels)
            modified_length = len(p_l)
            while modified_length != required_length:
                p_l.pop(-1)
                modified_length = len(p_l)

        predicted_labels.append(copy.deepcopy(p_l))
        gold_labels.append(copy.deepcopy(t_l))
    return predicted_labels, gold_labels

directory = '../prompt_and_response/ner/'
files = os.listdir(directory)
files.sort()

for file in files:
    if '.csv' not in file:
        continue
    print(file)
    data = pd.read_csv(directory+file)
    preds, labels = data_process(file, data)
    print(compute_metrics(preds, labels))


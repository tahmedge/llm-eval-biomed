import pandas as pd, csv
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer


def convert_hoc_labels(lines):
    labels = []
    classes = ['tumor promoting inflammation', 'inducing angiogenesis', 'evading growth suppressors', 'resisting cell death', 'cellular energetics',
               'genomic instability and mutation', 'sustaining proliferative signaling', 'avoiding immune destruction',
               'activating invasion and metastasis', 'enabling replicative immortality', "empty"]
    for line in lines:
        labels.append([w.strip() for w in line.strip().split('|')])

    return MultiLabelBinarizer(classes=classes).fit_transform(labels)

def do_eval(preds, golden):
    preds = convert_hoc_labels(preds)
    golden = convert_hoc_labels(golden)

    score = f1_score(golden, preds, average='micro')
    print(score)
    return

types = ['chatgpt', 'bard', 'claude2', 'llama2']
directory = '../prompt_and_response/hoc/'
for type in types:
    filename = directory+"with_"+type+"_response_new_hoc_detailed_prompt.csv"
    df = pd.read_csv(filename)
    dic_labels = {}
    preds = []
    golden = []
    for index, row in df.iterrows():
        classes = ['tumor promoting inflammation', 'inducing angiogenesis', 'evading growth suppressors',
                   'resisting cell death', 'cellular energetics', 'genomic instability and mutation',
                   'genome instability and mutation', 'deregulating cellular energetic metabolism',
                   'sustaining proliferative signaling', 'avoiding immune destruction',
                   'activating invasion and metastasis', 'enabling replicative immortality', "empty"]
        golden.append(row['label'])
        labels = row['label'].split("|")

        chatgpt_response = str(row['response']).lower().strip()
        chatgpt_response = chatgpt_response.split("[/inst]")[-1]

        if chatgpt_response[-1] == '.':
            chatgpt_response = chatgpt_response[:-1]
        annotation = 0

        for label in labels:
            dic_labels[label] = 1
            lowered_label = label.lower()
            track = 0
            if label == "empty":
                for cls in classes:
                    if cls in chatgpt_response:
                        if cls == "genome instability and mutation":
                            cls = 'genomic instability and mutation'
                        if cls == "deregulating cellular energetic metabolism":
                            cls = 'cellular energetics'
                        track = 1
                if track == 0:
                    chatgpt_response = "empty"
                if "empty" in chatgpt_response:
                    preds.append("empty")
                    annotation = 1
                    break
            else:
                if lowered_label in chatgpt_response:
                    preds.append(lowered_label)
                    annotation = 1
                    break

        if annotation == 0:

            track = 0
            for cls in classes:
                if cls in chatgpt_response:
                    if cls == "genome instability and mutation":
                        cls = 'genomic instability and mutation'
                    if cls == "deregulating cellular energetic metabolism":
                        cls = 'cellular energetics'
                    preds.append(cls)
                    track = 1
                    if lowered_label in cls:
                        annotation = 1
                    break
            if track == 0:
               preds.append("empty")

    print(type)
    do_eval(preds, golden)
    print()

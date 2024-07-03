import pandas as pd, os
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

types = ['bard', 'claude2', 'llama2']
directory = '../prompt_and_response/litcovid/'
files = os.listdir(directory)


def convert_hoc_labels(lines):
    labels = []
    classes = ['prevention', 'treatment', 'diagnosis', 'mechanism', 'case_report', 'transmission', 'forecasting', 'general']
    for line in lines:
        labels.append([w.strip() for w in line.strip().split('|')])
    return MultiLabelBinarizer(classes=classes).fit_transform(labels)

def do_eval(preds, golden):
    preds = convert_hoc_labels(preds)
    golden = convert_hoc_labels(golden)
    score = f1_score(golden, preds, average='micro')
    print(score)
    print()
    return


for file in files:
    dic_labels = {}
    preds = []
    golden = []
    filename = directory + file
    df = pd.read_csv(filename)
    print(filename)
    for index, row in df.iterrows():
        classes = ['Prevention', 'Treatment', 'Diagnosis', 'Mechanism', 'Case_Report', 'Transmission', 'Forecasting', 'General']
        row['label'] = str(row['label'])
        if row['label'] == 'nan':
            row['label'] = 'general'
        row['label'] = row['label'].replace(",", "|")
        row['label'] = row['label'].replace(" ", "")
        golden.append(str(row['label']).lower())

        labels = row['label'].strip().split("|")
        response = str(row['response']).lower().strip()
        response = response.split("[/inst]")[-1]
        response = response.replace("case report", "case_report")
        response = response.replace("casereport", "case_report")
        response = response.replace("(i)","prevention")
        response = response.replace("(ii)", "treatment")
        response = response.replace("(iii)", "diagnosis")
        response = response.replace("(iv)", "mechanism")
        response = response.replace("(v)", "case_report")
        response = response.replace("(vi)", "transmission")
        response = response.replace("(vii)", "forecasting")
        response = response.replace("(viii)", "general")
        count = 0
        for cls in classes:
            if cls.lower() in response:
                count += 1
                for label in labels:
                    if label.lower() == cls.lower():
                        break
        annotation = 0
        if count > 1:
            preds.append("general")
            annotation = 0
            continue

        annotation = 0
        for label in labels:
            dic_labels[label] = 1
            lowered_label = label.lower()
            track = 0
            if label == "general":
                for cls in classes:
                    if cls in response:
                        track = 1
                if track == 0:
                    response = "general"
                if "general" in response:
                    preds.append("general")
                    annotation = 1
                    break
            else:
                if lowered_label in response:
                    preds.append(lowered_label)
                    annotation = 1
                    break

        if annotation == 0:
            track = 0
            for cls in classes:
                if cls in response:
                    preds.append(cls)
                    track = 1
                    if lowered_label in cls:
                        annotation = 1
                    break

            if track == 0:
               preds.append("general")
    do_eval(preds, golden)

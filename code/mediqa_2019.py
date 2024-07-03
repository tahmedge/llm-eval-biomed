import csv
import pandas as pd
directory = '../prompt_and_response/mediqa/'
datasets = ['with_bard_response_pairwise_classification_mediqa2019_task3_question_answering.csv',
            'with_chatgpt_response_pairwise_classification_mediqa2019_task3_question_answering.csv',
            'with_llama_2_response_pairwise_classification_mediqa2019_task3_question_answering.csv',
            'with_claude_2_response_pairwise_classification_mediqa2019_task3_question_answering.csv'
            ]
for dataset in datasets:
    df = pd.read_csv(directory + dataset)
    match = 0
    unmatch = 0
    write_dataset = dataset.replace("_task3_question_answering","")
    write_dataset = write_dataset.replace("bard", "PaLM-2")
    write_dataset = write_dataset.replace("chatgpt", "GPT-3.5")
    write_dataset = write_dataset.replace("claude-2", "Claude-2")
    with open(write_dataset, "w+") as output_file:
        output_writer = csv.writer(output_file)
        output_writer.writerow(["gold_label", "predicted_label"])
        for index, row in df.iterrows():
            label = row['label_score']
            if label <= 2:
                label = 0
            else:
                label = 1
            if str(row['response'])=='nan':
                row['response'] = '0'
            row['response'] = str(row['response']).split("[/INST]")[-1]
            if str(label) in str(row['response']):
                match += 1
                if label == 0:
                    output_writer.writerow(["Not Relevant", "Not Relevant"])
                else:
                    output_writer.writerow(["Relevant", "Relevant"])
            else:
                unmatch += 1
                if label == 0:
                    output_writer.writerow(["Not Relevant", "Relevant"])
                else:
                    output_writer.writerow(["Relevant", "Not Relevant"])
        print(dataset, float(match/(match+unmatch))*100)
        print()


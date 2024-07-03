import evaluate, glob, pandas as pd
rouge = evaluate.load('rouge')
bertscore = evaluate.load('bertscore')
paths = ['../prompt_and_response/summarization/answer_summarization/',
'../prompt_and_response/summarization/dialogue_summarization/',
'../prompt_and_response/summarization/lay_summarization/',
'../prompt_and_response/summarization/question_summarization/']

for path in paths:
    all_files = glob.glob(path + "*.csv")
    print(all_files)
    all_dataframes = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        all_dataframes.append([filename, df])

    for data in all_dataframes:
        filename = data[0]
        df = data[1]
        predictions = []
        references = []
        print(filename)
        for index, row in df.iterrows():
            if 'chatgpt_response' in row:
                response = str(row['chatgpt_response'])
            elif 'bard_response' in row:
                response = str(row['bard_response'])
            else:
                response = str(row['response'])

            if "llama" in filename:
                response = response.split("[/INST]")[-1]
            
            predictions.append(str(response))
            if 'summary' in row:
                references.append(row['summary'])
            elif 'label' in row:
                references.append(row['label'])
        results_r = rouge.compute(predictions=predictions, references=references)
        results_b = bertscore.compute(predictions=predictions, references=references,
                                      model_type="roberta-large")
        print(results_r)
        print('BERTScore: ', sum(results_b['f1'])/len(results_b['f1']))

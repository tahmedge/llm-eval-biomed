import pandas as pd

types = ["bard", "chatgpt", "claude2","llama"]
directory = '../prompt_and_response/pubmedqa/'

for type in types:
    df = pd.read_csv(directory+"with_"+type+"_response_new_pubmedqa_test_with_question_context_answer_splitted_final.csv")
    match = 0
    total = 0
    for index, row in df.iterrows():
        label = row['label'].lower().strip()
        response = str(row['response']).lower()
        possible_labels = ["yes", "no", "maybe"]
        track = 0

        if type == "llama":
            response = response.split("[/inst]")[-1]

        if type == "claude2":
            if response.split(",")[0].strip() in possible_labels:
                response = response.split(",")[0]

        response = response.replace("\"", "")
        response = response.replace(".", " ")
        response = response.replace(",", " ")

        responses = response.split()

        formatted_response = []

        for res in responses:
            formatted_response.append(res.strip())

        formatted_response = list(set(formatted_response))

        for p_l in possible_labels:
            if p_l in formatted_response:
                track += 1

        if track <= 1 and label in response:
            match += 1
        total += 1
    print(type, match, total, match/total)


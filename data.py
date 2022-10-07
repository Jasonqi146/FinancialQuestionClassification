import torch

def open_file(csv_file):
    sentences = list()
    label1 = list()
    label2 = list()
    label3 = list()

    with open(csv_file) as file:
        try:
            header = file.readline()
            lines = file.readlines()

            for line in lines:
                _, sentence, rest = line.split("\"")
                _, l1, l2, l3 = rest.split(",")
                sentences.append(sentence)
                label1.append(l1)
                label2.append(l2)
                label3.append(l3)
        except:
            print(file)

    return sentences, label1, label2, label3

def read_files(path):
    import glob
    data_files = sorted(glob.glob(path + "*.csv"))

    t_sentences = list()
    t_labels = list()

    for file in data_files:
        sentences,label1, label2, _ = open_file(file)
        t_sentences += sentences
        t_labels += label1

    print("Finished reading images")
    return t_sentences, t_labels

def read_new_csv(path):
    import csv

    sentences = list()
    labels = list()

    with open(path) as f:
        reader = csv.reader(f)
        lines = list(reader)

        for sentence, label in lines:
            sentences.append(sentence.strip())
            labels.append(label.strip().split())

    return sentences, labels

def read_new_json(path):
    import json

    sentences = list()
    labels = list()


    with open(path) as f:
        lines = json.load(f)


        for sentence, label in lines:
            sentences.append(sentence.strip())
            labels.append(label)

    return sentences, labels

def get_data():
    t_sentences, t_labels = read_new_csv("question_annotation_2/questions_Eric.csv")
    t_sentences2, t_labels2 = read_new_json('question_annotation_2/final_labeling_task_jason_Nov29.json')
    
    # portion = int(4* len(t_sentences)/5)
    
    train_texts = t_sentences + t_sentences2
    print(len(train_texts))
    train_labels = t_labels + t_labels2
    
    # test_texts = t_sentences[portion:]
    # test_labels = t_labels[portion:]

    from sklearn.model_selection import train_test_split

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
    
    return train_texts, train_labels, val_texts, val_labels, val_texts, val_labels

label_to_num = {
    "Factoid" : 0,
    "Explanation" :1,
    "Casual": 2,
    "Opinion": 3,
    "Confirmation" : 4,
    "Hypothetical": 5,
    "Choice": 6,
    "Other": 7
}

LABELS = 8

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.zeros(LABELS)
        for label in self.labels[idx]:
            item['labels'][label_to_num[label]] = 1
        return item

    def __len__(self):
        return len(self.labels)

def get_dataset():
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_data()

    from transformers import DistilBertTokenizerFast

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    get_dataset()
from data import get_dataset
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_1 = DistilBertForSequenceClassification.from_pretrained('weights/distilbert')
model_1.to(device)
model_1.eval()

model_2 = DistilBertForSequenceClassification.from_pretrained('weights/distilbert_2')
model_2.to(device)
model_2.eval()

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

num_to_label = {
    0:"Factoid",
    1:"Explanation",
    2:"Casual",
    3:"Opinion",
    4:"Confirmation",
    5:"Hypothetical",
    6:"Choice",
    7:"Other"
}
THRESHOLD = 0.5

def classify(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    logit1 = model_1(**inputs)
    # output1 = torch.argmax(logit1[0], dim=1)
    pred_label = torch.sigmoid(logit1[0])
    pred_bool = [pl > THRESHOLD for pl in pred_label]

    import numpy as np
    labels = np.where(pred_bool == True)
    output1 = [num_to_label[item] for item in labels]

    return output1


if __name__ == '__main__':

    # print("Name sentence you wish to be classified: ")
    sentence =  "Hey, good morning, everybody, and thank you for taking my question. Not to belabor Galore Creek too much, but obviously over the last few years there has been a tremendous amount of uncertainty caused by resource nationalism and obviously Galore Creek safe jurisdiction in British Columbia. To what extent did that play a role in evaluation of this asset and the strategic decision to invest there? "
    output1 = classify(sentence)

    if len(output1) == 0:
        print("No label")
    for label in output1:
        print(label)

    #     input_ids = batch['input_ids'].to(device)
    #     attention_mask = batch['attention_mask'].to(device)
    #     labels = batch['labels'].to(device)
    #     loss, logits = model(input_ids, attention_mask=attention_mask, labels=labels)
    #
    #     import torch
    #     labels = torch.squeeze(labels)
    #     outputs = torch.argmax(logits, dim=1)
    #
    #     if labels.item() == outputs.item():
    #         correct_count += 1

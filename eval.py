from data import get_dataset
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

train_dataset, val_dataset, test_dataset = get_dataset()

from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = DistilBertForSequenceClassification.from_pretrained('weights/distilbert')
model.to(device)
model.eval()

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)


if __name__ == '__main__':
    print("Evaluating on {} questions".format(len(val_dataset)))
    correct_count = 0

    pred_labels = list()
    true_labels = list()
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask)

            b_logit_pred = logits[0]
            pred_label = torch.sigmoid(b_logit_pred)

            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            b_labels = labels.to('cpu').numpy()

            true_labels.append(b_labels)
            pred_labels.append(pred_label)

    # Flatten outputs
    pred_labels = [item for sublist in pred_labels for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]

    # Calculate Accuracy
    threshold = 0.50
    pred_bools = [pl > threshold for pl in pred_labels]
    true_bools = [tl == 1 for tl in true_labels]
    val_f1_accuracy = f1_score(true_bools, pred_bools, average='micro') * 100
    val_flat_accuracy = accuracy_score(true_bools, pred_bools) * 100

    val_recall = recall_score(true_bools, pred_bools,average='micro') * 100
    val_precision = precision_score(true_bools, pred_bools,average='micro') * 100

    print("Validation Precision: ", val_precision)
    print("Validation Recall: ", val_recall)
    print('F1 Validation Accuracy: ', val_f1_accuracy)
    print('Flat Validation Accuracy: ', val_flat_accuracy)


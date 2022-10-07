
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


if __name__ == '__main__':
    # classifier = pipeline('sentiment-analysis')
    #
    # model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    # pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    #
    # inputs = tokenizer("We are very happy to show you the 🤗 Transformers library.")
    #
    # pt_batch = tokenizer(
    #     ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    #     padding=True,
    #     truncation=True,
    #     return_tensors="pt"
    # )
    # for key, value in pt_batch.items():
    #     print(f"{key}: {value.numpy().tolist()}")
    #
    # pt_outputs = pt_model(**pt_batch)
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    import torch

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', return_dict=True)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
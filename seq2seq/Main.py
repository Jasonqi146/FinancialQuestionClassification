from seq2seq.Network import EncoderRNN, AttnDecoderRNN
from Format import input, output
from seq2seq.Train import trainIters
from seq2seq.Eval import evaluateRandomly
import torch

hidden_size = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    encoder1 = EncoderRNN(input.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output.n_words, dropout_p=0.1).to(device)

    trainIters(encoder1, attn_decoder1, 750, print_every=5, plot_every=5)

    evaluateRandomly(encoder1, attn_decoder1)
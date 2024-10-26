import torch
from model import Model, train, predict
from dataset import Dataset
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=32)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=4)
args = parser.parse_args()

dataset = Dataset(args)

if os.path.exists('model.pth'):
    model = torch.load("model.pth")
else:
    model = Model(dataset)

train(dataset, model, args)
torch.save(model, "model.pth")

pred = predict(dataset, model, text='I love', next_words=10)
pred = " ".join(pred)
print(pred)

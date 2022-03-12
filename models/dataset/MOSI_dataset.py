import os
import pickle
import numpy as np
import torch
import torch.utils.data as Data
from consts import GlobalConsts as gc

class MultimodalSubdata():
    def __init__(self, name="train"):
        self.name = name
        self.text = np.empty(0)
        self.audio = np.empty(0)
        self.video = np.empty(0)
        self.y = np.empty(0)

class MosiDataset(Data.Dataset):
    trainset = MultimodalSubdata("train")
    testset = MultimodalSubdata("test")
    validset = MultimodalSubdata("valid")

    def __init__(self, root, gc, clas="train"):
        self.root = root
        self.clas = clas
        if len(MosiDataset.trainset.y) != 0 and clas != "train":
            print("Data has been previously loaded, fetching from previous lists.")
        else:
            self.load_data(gc)

        if self.clas == "train":
            self.dataset = MosiDataset.trainset
        elif self.clas == "test":
            self.dataset = MosiDataset.testset
        elif self.clas == "valid":
            self.dataset = MosiDataset.validset

        self.text = self.dataset.text
        self.audio = self.dataset.audio
        self.video = self.dataset.video
        self.y = self.dataset.y


    def load_data(self,gc):
        if gc['proc_data'][-1] != '/':
            gc['proc_data'] = gc['proc_data'] + '/'
        dataset = pickle.load(open(gc['proc_data'] + 'mosi_data.pkl', 'rb'))
        video = 'video' if 'video' in dataset['test'] else 'vision'
        gc['padding_len'] = dataset['test']['text'].shape[1]
        gc['text_dim'] = dataset['test']['text'].shape[2]
        gc['audio_dim'] = dataset['test']['audio'].shape[2]
        gc['video_dim'] = dataset['test']['video'].shape[2]

        for ds, split_type in [(MosiDataset.trainset, 'train'), (MosiDataset.validset, 'valid'),
                               (MosiDataset.testset, 'test')]:
            ds.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
            ds.audio = torch.tensor(dataset[split_type]['audio'].astype(np.float32))
            ds.audio[ds.audio == -np.inf] = 0
            ds.audio = ds.audio.clone().cpu().detach()
            ds.video = torch.tensor(dataset[split_type]['video'].astype(np.float32)).cpu().detach()
            ds.y = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()
            ds.ids = dataset[split_type]['id'][:,0].reshape(-1)

    def __getitem__(self, index):
        inputLen = len(self.text[index])
        return self.text[index], self.audio[index], self.video[index], \
               inputLen, self.y[index].squeeze()

    def __len__(self):
        return len(self.y)


if __name__ == "__main__":
    dataset = MosiDataset(gc['proc_data'])
import torch
from torch.utils.data import Dataset
import os
import numpy as np

def string_to_board(s):
    count = int(0)
    for i in range(92):
        c = s[i]
        for j in range(4):
            me = (c >> (2*j)) & 1;
            you = (c >> (2*j + 1)) & 1;
            if me == 1:
                print(' o ', end='')
            elif you == 1:
                print(' x ', end='')
            else:
                print(' . ', end='')
            count += 1
            if count % 19 == 0:
                print("")
            if count >= 19*19:
                print("\n")
                return
    print("\n")

def board_to_numpy(s):
    data = np.zeros(shape=(19, 19), dtype=np.float)
    count = int(0)
    for i in range(92):
        c = s[i]
        for j in range(4):
            me = (c >> (2 * j)) & 1;
            you = (c >> (2 * j + 1)) & 1;
            if me == 1:
                data[int(count / 19)][int(count % 19)] = -1
            elif you == 1:
                data[int(count / 19)][int(count % 19)] = 1

            count += 1
            if count >= 19 * 19:
                return data
    return data

def print_board():
    with open("/home/javer/work/dataset/go/go.train", mode='rb') as f:
        for i in range(361):
            line = f.read(94)
            row = line[0]
            col = line[1]
            print(row, col)
            string_to_board(line[2:94])


class GoDataset(Dataset):
    def __init__(self, path="/home/javer/work/dataset/go/go.train"):
        self.path = path
        self.data = []
        with open("/home/javer/work/dataset/go/go.train", mode='rb') as f:
            self.data = f.read()


    def __len__(self):
        return int(os.path.getsize(self.path)/96)

    def __getitem__(self, index):
        data = self.data[index*94:(index*94+94)]
        row = data[0]
        col = data[1]
        state = data[2:94]
        label = row*19+col
        array_state = board_to_numpy(state)
        img = torch.from_numpy(array_state).float()
        img = img.reshape((1, 19, 19))
        return img, label

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


if __name__ == "__main__":
    dataset = GoDataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)
    a = 0
    for i in data_loader:
        a += 1
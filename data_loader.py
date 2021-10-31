import torch
from math import isclose
import typing


class DataLoader(torch.utils.data.Dataset):

    def __init__(self, features, targets):
        self.targets = torch.tensor(targets)
        self.features = torch.tensor(features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

    def __len__(self):
        return len(self.targets)

    def split(self, *split_ratio):
        '''
        returns torch random split object (this instance) \n
        if numbers > 1 are given: check if sum == len, then pass to torch.utils.data.random_split \n
        if numbers < 1 are given: then the sum must == 1, then compute the length og each portion \n
        '''
        summed = sum(split_ratio)
        if not isclose(summed, 1.0):  # or and abs( 1-sum(split_ratio) ) > 1e-15
            if summed != len(self):  # check if the sum == len
                raise ValueError(
                    f"sum of splitted lengths does not match the data length\ninstead of the total of {len(self)} or total ratio of 1, {split_ratio} -> {sum(split_ratio)} were given")
            self.sizes = split_ratio
        else:  # if sum == 1
            # leave one out, the left will be the last term, so that dun need to deal with the rounding problem
            self.sizes = [int(ratio*len(self)) for ratio in split_ratio[:-1]]
            # add the rest instead of the length*ratio, for the last term
            self.sizes.append(len(self) - sum(self.sizes))
        return torch.utils.data.random_split(self, self.sizes)

    def dataloader(self, *args, **kwargs: typing.Any):
        return torch.utils.data.DataLoader(self, *args, **kwargs)

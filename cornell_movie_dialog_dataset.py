from torch.utils.data import Dataset, DataLoader
import torch
import itertools
from prepare_data import PrepareData


class CornellMovieDialogDataset(Dataset):

    def __init__(self, config, paths, vocab, max_lengths):
        self.vocab = vocab
        self.max_lengths = max_lengths

        if isinstance(paths, str):
            paths = [paths]

        prepare_data = PrepareData(min_count=config.MIN_COUNT, max_length=config.MAX_LENGTH)
        parsed_pairs = sum([prepare_data.read_pairs(datafile) for datafile in paths], [])

        self.data = CornellMovieDialogDataset.make_dataset(parsed_pairs, vocab)

    @staticmethod
    def make_dataset(data, vocab):
        dataset = []
        for chat in data:
            chat1_ids = vocab.string2ids(chat[0])
            chat2_ids = vocab.string2ids(chat[1])
            dataset.append((chat1_ids, chat2_ids))
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chat_ids = self.data[idx]

        pairs = []
        #for i, chat_ids in enumerate(chats):
        ids0 = [self.vocab.bos_id] + chat_ids[0] + [self.vocab.eos_id]
        ids1 = [self.vocab.bos_id] + chat_ids[1] + [self.vocab.eos_id]

        #pairs.append((ids0, ids1))

        return (ids0, ids1)


    def collate_func(self, pair_batch):
        def zero_padding(l, fillvalue=self.vocab.pad_id):
            return list(itertools.zip_longest(*l, fillvalue=fillvalue))

        def binary_matrix(l, value_pad=self.vocab.pad_token):
            matrix = []

            for i, seq in enumerate(l):
                matrix.append([])
                for token in seq:
                    if token is value_pad:
                        matrix[i].append(0)
                    else:
                        matrix[i].append(1)

            return matrix

        # Returns padded input sequence tensor and lengths
        def input_var(ids_batch):
            lengths = torch.tensor([len(indexes) for indexes in ids_batch])
            pad_list = zero_padding(ids_batch)
            pad_var = torch.LongTensor(pad_list)
            return pad_var, lengths

        def output_var(ids_batch):
            max_target_length = max(len(indexes) for indexes in ids_batch)
            pad_list = zero_padding(ids_batch)
            mask = binary_matrix(pad_list)
            mask = torch.ByteTensor(mask)
            pad_var = torch.LongTensor(pad_list)
            return pad_var, mask, max_target_length

        pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
        input_batch, output_batch = [], []

        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])

        input_, lengths = input_var(input_batch)
        output, mask, max_target_length = output_var(output_batch)

        return input_, lengths, output, mask, max_target_length
from torch.utils.data import Dataset
import numpy as np
import torch as pt

class KGDataset_Uni(Dataset):
    def __init__(self, triples, entity_size, filter = False):
        super(KGDataset_Uni, self).__init__()
        self.data =triples
        self.entity_size = entity_size
        self.filter = filter
        if filter:
            self.data_as_set = set(tuple(triple) for triple in triples)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        pos_sample = self.data[index]
        neg_sample = pos_sample.copy()
        
        replace_index = 0 if np.random.rand() > 0.5 else 2
        neg_sample[replace_index] = np.random.randint(0, self.entity_size)
        while self.filter and (tuple(neg_sample) in self.data_as_set):
            replace_index = 0 if np.random.rand() > 0.5 else 2
            neg_sample[replace_index] = np.random.randint(0, self.entity_size)
        #print(pos_sample)
        return pos_sample, neg_sample
    
    @staticmethod
    def collate_fn(data):
        #print(data)
        pos_samples, neg_samples = zip(*data)
        pos_samples = pt.tensor(pos_samples, dtype=pt.long)
        neg_samples = pt.tensor(neg_samples, dtype=pt.long)
        return {'pos': pos_samples, 'neg': neg_samples}
    
class KGDataset_Bern(Dataset):
    def __init__(self, triples, entity_size, r2ht, filter = False):
        super(KGDataset_Bern, self).__init__()
        self.data =triples
        self.entity_size = entity_size
        self.filter = filter
        self.r2ht = r2ht
        if filter:
            self.data_as_set = set(tuple(triple) for triple in triples)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        pos_sample = self.data[index]
        neg_sample = pos_sample.copy()
        threshhold = len(self.r2ht[pos_sample[1]][0]) / (len(self.r2ht[pos_sample[1]][0]) + len(self.r2ht[pos_sample[1]][1]))
        replace_index = 0 if np.random.rand() < threshhold else 2
        neg_sample[replace_index] = np.random.randint(0, self.entity_size)
        while self.filter and (tuple(neg_sample) in self.data_as_set):
            replace_index = 0 if np.random.rand() < threshhold else 2
            neg_sample[replace_index] = np.random.randint(0, self.entity_size)
        #print(pos_sample)
        return pos_sample, neg_sample

    @staticmethod
    def collate_fn(data):
        #print(data)
        pos_samples, neg_samples = zip(*data)
        pos_samples = pt.tensor(pos_samples, dtype=pt.long)
        neg_samples = pt.tensor(neg_samples, dtype=pt.long)
        return {'pos': pos_samples, 'neg': neg_samples}

class KGDataset_Test_Chunked(Dataset):
    def __init__(self, chunk_data, entity_size):
        super(KGDataset_Test_Chunked, self).__init__()

        self.entity_size = entity_size

        self.data = chunk_data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        pos_sample = self.data[index]
        return pos_sample
    
    @staticmethod
    def collate_fn(data):
        pos_samples = pt.tensor(data, dtype=pt.long)
        return pos_samples

class KGDataset_Test(Dataset):
    def __init__(self, head_data, tail_data, entity_size):
        super(KGDataset_Test, self).__init__()

        self.entity_size = entity_size

        self.data = head_data + tail_data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        pos_sample = self.data[index]
        return pos_sample
    
    @staticmethod
    def collate_fn(data):
        pos_samples = pt.tensor(data, dtype=pt.long)
        return pos_samples


def load_data(data_path, verbose=True):
    dataset_division_names = ['train.txt', 'valid.txt', 'test.txt']
    rel2num = dict()
    entity2num = dict()
    num2entity = dict()
    num2rel = dict()
    r2ht = dict()
    triples = {'train.txt': [], 'valid.txt': [], 'test.txt': []}
    
    
    output_name = data_path.split('/')[-2]
    if verbose:
        print(output_name)
    for div_name in dataset_division_names:
        dataset_path = data_path + div_name
        if verbose:
            print(div_name)
        with open(dataset_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                h, r, t = line.split()
            
                if h not in entity2num:
                    h_num = len(entity2num)
                    entity2num[h] = h_num
                    num2entity[h_num] = h
                else:
                    h_num = entity2num[h]
                    
                if t not in entity2num:
                    t_num = len(entity2num)
                    entity2num[t] = t_num
                    num2entity[t_num] = t
                else:
                    t_num = entity2num[t]
                    
                if r not in rel2num:
                    r_num = len(rel2num)
                    rel2num[r] = r_num
                    num2rel[r] = r
                else:
                    r_num = rel2num[r]
                
                if r_num not in r2ht:
                    r2ht[r_num] = [{h_num}, {t_num}]
                else:
                    r2ht[r_num][0].add(h_num)
                    r2ht[r_num][1].add(t_num)
                
                triples[div_name].append([h_num, r_num, t_num])
    if verbose:
        print(f"Train/Dev/Test: {len(triples['train.txt'])}/{len(triples['valid.txt'])}/{len(triples['test.txt'])}")
        print(f'Entity Num: {len(entity2num)}')
        print(f'Relation Num: {len(rel2num)}')
    
    return triples['train.txt'], triples['valid.txt'], triples['test.txt'], entity2num, num2entity, rel2num, num2rel, r2ht 
    
def get_corrupted_head_test_data(rt_list, entity_num):
    for rt in rt_list:
        yield [[corrupted_head, rt[0], rt[1]] for corrupted_head in range(entity_num)]
        
def get_corrupted_tail_test_data(hr_list, entity_num):
    for hr in hr_list:
        yield [[hr[0], hr[1], corrupted_tail] for corrupted_tail in range(entity_num)]

def get_ordered_mapping_chunked(triples, entity_num):
    
    rt2h = dict()
    hr2t = dict()
    # rt2num = dict()
    # hr2num = dict()
    # num2rt = dict()
    # num2hr = dict()
    # head_data = set()
    # tail_data = set()
    
    for triple in triples:
        if (triple[0], triple[1]) not in hr2t:
            hr2t[(triple[0], triple[1])] = {triple[2]}
        else:
            hr2t[(triple[0], triple[1])].add(triple[2])
            
        if (triple[1], triple[2]) not in rt2h:
            rt2h[(triple[1], triple[2])] = {triple[0]}
        else:
            rt2h[(triple[1], triple[2])].add(triple[0])

    #     if tuple(triple) not in head_data:
    #         head_data.update([(neg_head, triple[1], triple[2]) for neg_head in range(entity_size)])
    #     if tuple(triple) not in tail_data:
    #         tail_data.update([(triple[0], triple[1], neg_tail) for neg_tail in range(entity_size)])
                
    # head_data  = sorted(list(head_data), key=lambda x: (x[1], x[2], x[0]))
    # tail_data  = sorted(list(tail_data), key=lambda x: (x[0], x[1], x[2]))
        
    # head_index = 0
    # for triple in head_data:
    #     if (triple[1], triple[2]) not in rt2num:
    #         rt2num[(triple[1], triple[2])] = head_index
    #         num2rt[head_index] = (triple[1], triple[2])
    #         head_index += 1
            
    # tail_index = len(rt2num)
    # for triple in tail_data:
    #     if (triple[0], triple[1]) not in hr2num:
    #         hr2num[(triple[0], triple[1])] = tail_index
    #         num2hr[tail_index] =  (triple[0], triple[1])
    #         tail_index += 1
    rt_list = sorted(rt2h.keys())
    hr_list = sorted(hr2t.keys())
    head_data_generator = get_corrupted_head_test_data(rt_list, entity_num)
    tail_data_generator = get_corrupted_tail_test_data(hr_list, entity_num)
                
    return head_data_generator, tail_data_generator, rt2h, hr2t


def get_ordered_mapping(triples, entity_size):
    
    rt2h = dict()
    hr2t = dict()
    rt2num = dict()
    hr2num = dict()
    num2rt = dict()
    num2hr = dict()
    head_data = set()
    tail_data = set()
    
    for triple in triples:
        if (triple[0], triple[1]) not in hr2t:
            hr2t[(triple[0], triple[1])] = {triple[2]}
        else:
            hr2t[(triple[0], triple[1])].add(triple[2])
            
        if (triple[1], triple[2]) not in rt2h:
            rt2h[(triple[1], triple[2])] = {triple[0]}
        else:
            rt2h[(triple[1], triple[2])].add(triple[0])
            
    for triple in triples:
        if tuple(triple) not in head_data:
            head_data.update([(neg_head, triple[1], triple[2]) for neg_head in range(entity_size)])
        if tuple(triple) not in tail_data:
            tail_data.update([(triple[0], triple[1], neg_tail) for neg_tail in range(entity_size)])
    head_data  = sorted(list(head_data), key=lambda x: (x[1], x[2], x[0]))
    tail_data  = sorted(list(tail_data), key=lambda x: (x[0], x[1], x[2]))
        
    head_index = 0
    for triple in head_data:
        if (triple[1], triple[2]) not in rt2num:
            rt2num[(triple[1], triple[2])] = head_index
            num2rt[head_index] = (triple[1], triple[2])
            head_index += 1
            
    tail_index = len(rt2num)
    for triple in tail_data:
        if (triple[0], triple[1]) not in hr2num:
            hr2num[(triple[0], triple[1])] = tail_index
            num2hr[tail_index] =  (triple[0], triple[1])
            tail_index += 1
            
    return head_data, tail_data, rt2h, hr2t, hr2num, rt2num, num2hr, num2rt

        
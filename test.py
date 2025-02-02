from TransX import TransE, TransH
from utils import load_data, get_ordered_mapping, KGDataset_Uni, KGDataset_Test
import torch as pt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import SGD


if __name__ == '__main__':
    
    device = 'cuda' if pt.cuda.is_available() else 'cpu'
    print(device)
    emb_size = 50
    data_path = './data/FB15K/'
    save_name = data_path.split('/')[-2]
    train_batch_size = 1200
    dev_batch_size = 3600
    epochs = 1
    res_norm = 1
    margin = 1
    lr = 0.01
    C = 0.015625
    logging_step = 50
    logging_loss = 0
    filter = False
    debug = False
    logging_dir = './log'
    model_dir = '/local/data/zgd324'
    sample_mode = 'Uni' # 'Uni or Bern'
    model_name = 'TransE' # or 'TransH'
    run_mode = 'dev'
    chunked = False
    save_name = save_name + f"_{model_name}_{sample_mode}_{'filter' if filter else 'wo_filter'}_{run_mode}_{C}"
    
    train_data, val_data, test_data, entity2num, num2entity, rel2num, num2rel, r2ht  = load_data(data_path)
    if model_name == 'TransE':
        model = TransE(len(entity2num), len(rel2num), emb_size)
    elif model_name == 'TransH':
        model = TransH(len(entity2num), len(rel2num), emb_size, C)
    else:
        raise NotImplementedError
    model.to(device)
    
    #print(val_data)
    res_f = open(f'{logging_dir}/{save_name}_res.txt', 'w')
    

        
    model = pt.load(f'{model_dir}/{save_name}_{500}.pt')
    model.to(device)
    print('Eval')
    model.eval()
    res_dict = {}
    with pt.no_grad():
        scores = []
        indices2rank_list = []
        rank_sum = 0
        hit10 = 0
        head_data_generator, tail_data_generator, rt2h, hr2t = get_ordered_mapping(test_data, len(entity2num))
        for rt, head_data in zip(sorted(rt2h.keys()), head_data_generator):
            score_per_rt = []
            dev_dataset = KGDataset_Test(head_data , len(entity2num)) #后处理一下filter设定
            dev_datalaoder = DataLoader(dev_dataset, dev_batch_size, shuffle=False, collate_fn=KGDataset_Test.collate_fn)
            for step, batch in enumerate(dev_datalaoder):
                h, r, t = batch[ : , 0], batch[ : , 1], batch[ : , 2]
                h = h.to(device)
                r = r.to(device)
                t = t.to(device)
                dissim_score = model.forward(h, r, t, res_p = res_norm)
                #print(dissim_score)
                score_per_rt.append(dissim_score)
            score_per_rt = pt.cat(score_per_rt)
            _, indices = pt.sort(score_per_rt)
            indices2rank = dict(zip(indices.tolist(), list(range(1, len(indices) + 1))))
            if filter:
                pass
            else:
                    h_set = rt2h[rt]
                    for h in h_set:
                        rank = indices2rank[h]
                        rank_sum += rank
                        if rank <= 10:
                            hit10 += 1
        for hr, tail_data in zip(sorted(hr2t.keys()), tail_data_generator):
            score_per_hr = []
            dev_dataset = KGDataset_Test(tail_data , len(entity2num)) #后处理一下filter设定
            dev_datalaoder = DataLoader(dev_dataset, dev_batch_size, shuffle=False, collate_fn=KGDataset_Test.collate_fn)
            print(len(dev_datalaoder))
            for step, batch in enumerate(dev_datalaoder):
                if step % 500 == 0:
                    print(f'{step}/{len(dev_datalaoder)}')
                h, r, t = batch[ : , 0], batch[ : , 1], batch[ : , 2]
                h = h.to(device)
                r = r.to(device)
                t = t.to(device)
                dissim_score = model.forward(h, r, t, res_p = res_norm)
                #print(dissim_score)
                score_per_hr.append(dissim_score)
            score_per_hr = pt.cat(score_per_hr)
            _, indices = pt.sort(score_per_hr)
            indices2rank = dict(zip(indices.tolist(), list(range(1, len(indices) + 1))))                                
            if filter:
                pass
            else:
                    t_set = hr2t[hr]
                    for t in t_set:
                        rank = indices2rank[t]
                        rank_sum += rank
                        if rank <= 10:
                            hit10 += 1
                            
        mean_rank = rank_sum / len(val_data) / 2 #tail and head are both replaced so twice
        hit10 = hit10 / len(val_data) / 2 
        
        res_f.write(f'MR: {mean_rank}\n')
        res_f.write(f'hit10: {hit10}\n')
        res_f.close()
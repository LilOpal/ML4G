from TransX import TransE, TransH
from utils import load_data, get_ordered_mapping, get_ordered_mapping_chunked, KGDataset_Uni, KGDataset_Bern, KGDataset_Test, KGDataset_Test_Chunked
import torch as pt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import SGD
from tqdm import tqdm


if __name__ == '__main__':
    
    device = 'cuda' if pt.cuda.is_available() else 'cpu'
    print(pt.version.cuda)
    print(device)
    emb_size = 50
    data_path = './datasets_knowledge_embedding/WN18/'
    save_name = data_path.split('/')[-2]
    train_batch_size = 1200
    dev_batch_size = 1200
    epochs = 10
    res_norm = 1
    margin = 1
    lr = 0.01
    C = 0.001
    logging_step = 10
    logging_loss = 0
    filter = False
    debug = False
    sample_mode = 'Bern' # 'Uni or Bern'
    model_name = 'TransH' # 'TransH' or 'TransE'
    run_mode = 'train'
    chunked = True
    save_name = save_name + f"_{model_name}_{sample_mode}_{'filter' if filter else 'wo_filter'}_{run_mode}"
    
    
    train_data, val_data, test_data, entity2num, num2entity, rel2num, num2rel, r2ht  = load_data(data_path)
    #print(r2ht)
    if run_mode == 'dev':
        val_data = train_data
    if model_name == 'TransE':
        model = TransE(len(entity2num), len(rel2num), emb_size)
    elif model_name == 'TransH':
        model = TransH(len(entity2num), len(rel2num), emb_size, C)
    else:
        raise NotImplementedError
    model.to(device)
    
    if sample_mode == 'Bern':
        train_dataset = KGDataset_Bern(train_data, len(entity2num), r2ht, filter)
    elif sample_mode == 'Uni':
        train_dataset = KGDataset_Uni(train_data, len(entity2num), filter)
    train_dataloader = DataLoader(train_dataset, train_batch_size, shuffle=True, collate_fn=KGDataset_Bern.collate_fn)
    
    #print(val_data)

    
    optimizer = SGD(model.parameters(), lr = lr)
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            #print(batch['pos'])
            h, r, t = batch['pos'][ : , 0], batch['pos'][ : , 1], batch['pos'][ : , 2]
            h = h.to(device)
            r = r.to(device)
            t = t.to(device)
            pos_score = model.forward(h, r, t, res_p = res_norm)
            neg_h, neg_r, neg_t = batch['neg'][ : , 0], batch['neg'][ : , 1], batch['neg'][ : , 2]
            neg_h = neg_h.to(device)
            neg_r = neg_r.to(device)
            neg_t = neg_t.to(device)
            neg_score = model.forward( neg_h, neg_r, neg_t, res_p = res_norm)
            
            loss = F.relu(pos_score - neg_score + margin).sum() 
            loss.backward()
            logging_loss += loss.item()
            
            if (step + 1) % logging_step == 0:
                logging_loss /= logging_step
                print(logging_loss)
                logging_loss = 0
            
            optimizer.step()
        print(epoch)
    print('Eval')
    model.eval()
    res_dict = {}
    with pt.no_grad():
        scores = []
        indices2rank_list = []
        rank_sum = 0
        hit10 = 0
        
        rank_sum_filtered = 0
        hit10_filtered = 0
        
        if chunked:
            head_data_generator, tail_data_generator, rt2h, hr2t = get_ordered_mapping_chunked(test_data, len(entity2num))
            for rt, head_data in zip(sorted(rt2h.keys()), head_data_generator):
                score_per_rt = []
                dev_dataset = KGDataset_Test_Chunked(head_data , len(entity2num)) #后处理一下filter设定
                dev_datalaoder = DataLoader(dev_dataset, dev_batch_size, shuffle=False, collate_fn=KGDataset_Test.collate_fn)
                for step, batch in enumerate(dev_datalaoder):
                    h, r, t = batch[ : , 0], batch[ : , 1], batch[ : , 2]
                    h = h.to(device)
                    r = r.to(device)
                    t = t.to(device)
                    dissim_score = model.forward(h, r, t, res_p = res_norm, train=False)
                    #print(dissim_score)
                    score_per_rt.append(dissim_score)
                score_per_rt = pt.cat(score_per_rt)
                _, indices = pt.sort(score_per_rt)
                indices2rank = dict(zip(indices.tolist(), list(range(1, len(indices) + 1))))
                
                head_rank_list = []
                h_set = rt2h[rt]
                for h in h_set:
                    rank = indices2rank[h]
                    rank_sum += rank
                    if rank <= 10:
                        hit10 += 1
                    head_rank_list.append(rank)
                    
                head_rank_list.sort()
                for n_before, h_rank in enumerate(head_rank_list):
                    corrected_h_rank = rank - n_before
                    rank_sum_filtered += corrected_h_rank
                    if corrected_h_rank <= 10:
                        hit10_filtered += 1
                
            for hr, tail_data in zip(sorted(hr2t.keys()), tail_data_generator):
                score_per_hr = []
                dev_dataset = KGDataset_Test_Chunked(tail_data , len(entity2num)) #后处理一下filter设定
                dev_datalaoder = DataLoader(dev_dataset, dev_batch_size, shuffle=False, collate_fn=KGDataset_Test.collate_fn)
                for step, batch in enumerate(dev_datalaoder):
                    h, r, t = batch[ : , 0], batch[ : , 1], batch[ : , 2]
                    h = h.to(device)
                    r = r.to(device)
                    t = t.to(device)
                    dissim_score = model.forward(h, r, t, res_p = res_norm, train=False)
                    #print(dissim_score)
                    score_per_hr.append(dissim_score)
                score_per_hr = pt.cat(score_per_hr)
                _, indices = pt.sort(score_per_hr)
                indices2rank = dict(zip(indices.tolist(), list(range(1, len(indices) + 1))))                                
                
                tail_rank_list = []
                t_set = hr2t[hr]
                for t in t_set:
                    rank = indices2rank[t]
                    rank_sum += rank
                    #print(rank)
                    if rank <= 10:
                        hit10 += 1
                    tail_rank_list.append(rank)    
                    
                tail_rank_list.sort()
                for n_before, t_rank in enumerate(tail_rank_list):
                    corrected_t_rank = t_rank - n_before
                    rank_sum_filtered += corrected_t_rank
                    if corrected_t_rank <= 10:
                        hit10_filtered += 1  
        else:
            # print(val_data)
            print('eval')
            head_data, tail_data, rt2h, hr2t, hr2num, rt2num, num2hr, num2rt = get_ordered_mapping(test_data, len(entity2num))
            print('eval')
            dev_dataset = KGDataset_Test(head_data, tail_data, len(entity2num)) #后处理一下filter设定
            print('eval')
            dev_dataloader = DataLoader(dev_dataset, dev_batch_size, shuffle=False, collate_fn=KGDataset_Test.collate_fn)
            print('eval')
            print(len(dev_dataloader))
            for step, batch in enumerate(dev_dataloader):
                h, r, t = batch[ : , 0], batch[ : , 1], batch[ : , 2]
                h = h.to(device)
                r = r.to(device)
                t = t.to(device)
                dissim_score = model.forward(h, r, t, res_p = res_norm, train=False)
                #print(dissim_score)
                scores.append(dissim_score)
            scores = pt.cat(scores)
            scores = scores.chunk(len(scores) // len(entity2num))
            
            #pt.save(scores, f'./log/{save_name}_scores.pt')
            for score in scores:
                _, indices = pt.sort(score)
                indices2rank = dict(zip(indices.tolist(), list(range(1, len(indices) + 1))))
                indices2rank_list.append(indices2rank)
            #print(len(indices2rank_list), len(rt2h), len(hr2t))
            assert len(indices2rank_list) == len(rt2h) + len(hr2t)
            
            for rt in rt2h.keys():
                head_rank_list = []
                h_set = rt2h[rt]
                index = rt2num[rt]
                head_indices2rank = indices2rank_list[index]
                for h in h_set:
                    rank = head_indices2rank[h]
                    rank_sum += rank
                    if rank <= 10:
                        hit10 += 1
                    head_rank_list.append(rank)
                
                head_rank_list.sort()
                for n_before, h_rank in enumerate(head_rank_list):
                    corrected_h_rank = rank - n_before
                    rank_sum_filtered += corrected_h_rank
                    if corrected_h_rank <= 10:
                        hit10_filtered += 1
                        
            for hr in hr2t.keys():
                tail_rank_list = []
                t_set = hr2t[hr]
                index = hr2num[hr]
                tail_indices2rank = indices2rank_list[index]
                for t in t_set:
                    rank = tail_indices2rank[t]
                    rank_sum += rank
                    if rank <= 10:
                        hit10 += 1
                    tail_rank_list.append(rank)
                    
                tail_rank_list.sort()
                for n_before, t_rank in enumerate(tail_rank_list):
                    corrected_t_rank = t_rank - n_before
                    rank_sum_filtered += corrected_t_rank
                    if corrected_t_rank <= 10:
                        hit10_filtered += 1    
                
                
        mean_rank = rank_sum / len(test_data) / 2 #tail and head are both replaced so twice
        hit10 = hit10 / len(test_data) / 2 
        mean_rank_filtered = rank_sum_filtered / len(test_data) / 2
        hit10_filtered = hit10_filtered / len(test_data) / 2
    
        print(f'MR/MR_filt: {mean_rank}/{mean_rank_filtered}')
        print(f'hit10/hit10_filt: {hit10}/{hit10_filtered}')
 
        # model.eval()
        # res_dict = {}
        # with pt.no_grad():
        #     scores = []
        #     indices2rank_list = []
        #     rank_sum = 0
        #     hit10 = 0

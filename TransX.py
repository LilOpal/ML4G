import torch as pt
import torch.nn as nn

class TransE(nn.Module):
    def __init__(self, input_entity_size, input_rel_size, emb_size):
        super(TransE, self).__init__()
        self.input_entity_size = input_entity_size
        self.input_rel_size = input_rel_size
        self.emb_size = emb_size
        
        self.emb_entity  = nn.Embedding(input_entity_size, emb_size)
        self.emb_relation = nn.Embedding(input_rel_size, emb_size)
        nn.init.xavier_uniform_(self.emb_entity.weight.data)
        nn.init.xavier_uniform_(self.emb_relation.weight.data)
        
    def get_emb_unnorm(self, h, r, t):
        
        h_emb = self.emb_entity(h)
        r_emb = self.emb_relation(r)
        t_emb = self.emb_entity(t)
        
        return h_emb, r_emb, t_emb
    
    def get_emb_norm(self, h, r, t, norm_p = 2):
        
        h_emb, r_emb, t_emb = self.get_emb_unnorm(h, r, t)
        
        h_emb = nn.functional.normalize(h_emb, p = norm_p, dim = -1)
        r_emb = nn.functional.normalize(r_emb, p = norm_p, dim = -1)
        t_emb = nn.functional.normalize(t_emb, p = norm_p, dim = -1)
        
        return h_emb, r_emb, t_emb
    
    def forward(self, h, r, t, norm_p = 2, res_p = 2, train=False):
        
        h_emb, r_emb, t_emb = self.get_emb_norm(h, r, t, norm_p)
        
        return pt.norm(h_emb - t_emb + r_emb, p = res_p, dim=-1)
        
        

class TransH(nn.Module):
    def __init__(self, input_entity_size, input_rel_size, emb_size, C):
        super(TransH, self).__init__()
        self.input_entity_size = input_entity_size
        self.input_rel_size = input_rel_size
        self.emb_size = emb_size
        
        self.emb_entity  = nn.Embedding(input_entity_size, emb_size)
        self.emb_relation = nn.Embedding(input_rel_size, emb_size)
        self.emb_normal_vec = nn.Embedding(input_rel_size, emb_size)
        self.C = C
        nn.init.xavier_uniform_(self.emb_entity.weight.data)
        nn.init.xavier_uniform_(self.emb_relation.weight.data)
        nn.init.xavier_uniform_(self.emb_normal_vec.weight.data)
        
    def get_emb_unnorm(self, h, r, t):
        
        h_emb = self.emb_entity(h)
        r_emb = self.emb_relation(r)
        t_emb = self.emb_entity(t)
        
        return h_emb, r_emb, t_emb
    
    def get_emb_norm(self, h, r, t, norm_p = 2):
        
        h_emb, r_emb, t_emb = self.get_emb_unnorm(h, r, t)
        
        h_emb = nn.functional.normalize(h_emb, p = norm_p, dim = -1)
        r_emb = nn.functional.normalize(r_emb, p = norm_p, dim = -1)
        t_emb = nn.functional.normalize(t_emb, p = norm_p, dim = -1)
        
        return h_emb, r_emb, t_emb
    
    @staticmethod
    def proj(emb, norm_vec):

        return emb - (norm_vec * emb).sum(dim = -1, keepdim=True) * norm_vec
    
    @staticmethod
    def scale_constrain(h, t):
        return pt.nn.functional.relu((h * h).sum() - 1) + pt.nn.functional.relu((t * t).sum() - 1)
    
    def forward(self, h, r, t, norm_p = 2, res_p = 2, train = True):
        
        h_emb, r_emb, t_emb = self.get_emb_unnorm(h, r, t)
        normal_emb  = self.emb_normal_vec(r)
        normal_emb = nn.functional.normalize(normal_emb, p = 2, dim=-1)
        h_emb = self.proj(h_emb, normal_emb)
        t_emb = self.proj(t_emb, normal_emb)
        if self.C >= 0 and train:
            return pt.norm(h_emb - t_emb + r_emb, p = res_p, dim=-1) + self.C * (nn.functional.relu((normal_emb * r_emb).sum(dim = 1) - 1e-6).sum() + self.scale_constrain(h_emb, t_emb))
        else:
            return pt.norm(h_emb - t_emb + r_emb, p = res_p, dim=-1)
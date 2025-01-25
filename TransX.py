import torch as pt
import torch.nn as nn

class TransE(nn.Module):
    def __init__(self, input_entity_size, input_rel_size, emb_size):
        self.input_entity_size = input_entity_size
        self.input_rel_size = input_rel_size
        self.emb_size = emb_size
        
        self.emb_entity  = nn.Embedding(input_entity_size, emb_size)
        self.emb_relation = nn.Embedding(input_rel_size, emb_size)
        
    def get_emb_unnorm(self, h, r, t):
        
        h_emb = self.emb_entity(h)
        r_emb = self.emb_relation(r)
        t_emb = self.emb_entity(t)
        
        return h_emb, r_emb, t_emb
    
    def get_emb_norm(self, h, r, t, norm_p = 2):
        
        h_emb, r_emb, t_emb = self.get_emb_unnorm(h, r, t)
        
        h_emb = pt.norm(h_emb, p = norm_p, dim = -1)
        r_emb = pt.norm(r_emb, p = norm_p, dim = -1)
        t_emb = pt.norm(t_emb, p = norm_p, dim = -1)
        
        return h_emb, r_emb, t_emb
    
    def forward(self, h, r, t, norm_p = 2):
        
        h_emb, r_emb, t_emb = self.get_emb_norm(h, r, t, norm_p)
        
        return pt.norm(h_emb - t_emb + r_emb, p = 2, dim=-1)
        
        

class TransH(nn.Module):
    def __init__(self, input_entity_size, input_rel_size, emb_size):
        self.input_entity_size = input_entity_size
        self.input_rel_size = input_rel_size
        self.emb_size = emb_size
        
        self.emb_entity  = nn.Embedding(input_entity_size, emb_size)
        self.emb_relation = nn.Embedding(input_rel_size, emb_size)
        self.emb_normal_vec = nn.Embedding(input_rel_size, emb_size)
        
    def get_emb_unnorm(self, h, r, t):
        
        h_emb = self.emb_entity(h)
        r_emb = self.emb_relation(r)
        t_emb = self.emb_entity(t)
        
        return h_emb, r_emb, t_emb
    
    def get_emb_norm(self, h, r, t, norm_p = 2):
        
        h_emb, r_emb, t_emb = self.get_emb_unnorm(h, r, t)
        
        h_emb = pt.norm(h_emb, p = norm_p, dim = -1)
        r_emb = pt.norm(r_emb, p = norm_p, dim = -1)
        t_emb = pt.norm(t_emb, p = norm_p, dim = -1)
        
        return h_emb, r_emb, t_emb
    
    @staticmethod
    def proj(emb, norm_vec):
        return emb - pt.dot(norm_vec, emb) * norm_vec
    
    def forward(self, h, r, t, norm_p = 2):
        
        h_emb, r_emb, t_emb = self.get_emb_unnorm(h, r, t, norm_p)
        normal_emb  = self.emb_normal_vec(r)
        normal_emb = pt.norm(normal_emb, p = 2, dim=-1)
        h_emb = self.proj(h_emb, normal_emb)
        t_emb = self.proj(t_emb, normal_emb)
        
        return pt.norm(h_emb - t_emb + r_emb, p = 2, dim=-1)
    
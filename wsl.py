import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict, Counter
from utils import compute_uncertainty, solve_sinkhorn_pot

class CMA(nn.Module):
    '''
    Cross modal Match Aggregation (Repaired for High Uncertainty Robustness)
    '''
    def __init__(self, args):
        super(CMA, self).__init__()
        self.device = torch.device(args.device)
        self.not_saved = True
        self.num_classes = args.num_classes
        self.sigma = args.sigma 
        
        self.ot_alpha = getattr(args, 'ot_alpha', 0.01) # Default lower alpha
        self.ot_reg = getattr(args, 'ot_reg', 0.02)     # Default sharper reg
        self.ot_mass = getattr(args, 'ot_mass', 0.8)
        
        self.register_buffer('vis_memory', torch.zeros(self.num_classes, 2048))
        self.register_buffer('ir_memory', torch.zeros(self.num_classes, 2048))

        self.saved_rgb_feats = None
        self.saved_ir_feats = None
        self.saved_rgb_logits = None
        self.saved_ir_logits = None
        self.saved_rgb_ids = None
        self.saved_ir_ids = None

    @torch.no_grad()
    def update(self, rgb_features, ir_features, rgb_ids, ir_ids):
        self.vis_memory = self.vis_memory.to(self.device)
        self.ir_memory = self.ir_memory.to(self.device)
        
        rgb_label_set = torch.unique(rgb_ids)
        for label in rgb_label_set:
            mask = (rgb_ids == label)
            if mask.any():
                proto = rgb_features[mask].mean(dim=0)
                if self.vis_memory[label].abs().sum() == 0:
                     self.vis_memory[label] = proto
                else:
                    self.vis_memory[label] = (1 - self.sigma) * self.vis_memory[label] + self.sigma * proto

        ir_label_set = torch.unique(ir_ids)
        for label in ir_label_set:
            mask = (ir_ids == label)
            if mask.any():
                proto = ir_features[mask].mean(dim=0)
                if self.ir_memory[label].abs().sum() == 0:
                    self.ir_memory[label] = proto
                else:
                    self.ir_memory[label] = (1 - self.sigma) * self.ir_memory[label] + self.sigma * proto

    @torch.no_grad()
    def save(self, rgb_logits, ir_logits, rgb_ids, ir_ids, rgb_feats, ir_feats):
        self.not_saved = False
        self.update(rgb_feats, ir_feats, rgb_ids, ir_ids)
        self.saved_rgb_feats = rgb_feats
        self.saved_ir_feats = ir_feats
        self.saved_rgb_logits = rgb_logits
        self.saved_ir_logits = ir_logits
        self.saved_rgb_ids = rgb_ids
        self.saved_ir_ids = ir_ids
        
    @torch.no_grad()
    def get_label(self, epoch=None):
        if self.not_saved:
            return {}, {}

        rgb_f = torch.nn.functional.normalize(self.saved_rgb_feats, p=2, dim=1)
        ir_f = torch.nn.functional.normalize(self.saved_ir_feats, p=2, dim=1)
        
        sim_mat = torch.matmul(rgb_f, ir_f.t())
        dist_mat = 1.0 - sim_mat
        
        # Debug Log
        print(f"[CMA] Dist Stats: Min={dist_mat.min():.4f}, Mean={dist_mat.mean():.4f}")
        
        # Uncertainty
        u_rgb = compute_uncertainty(self.saved_rgb_logits)
        u_ir = compute_uncertainty(self.saved_ir_logits)
        
        # New logging to confirm fix
        print(f"[CMA] Uncertainty (Absolute): RGB_Mean={u_rgb.mean():.4f}, IR_Mean={u_ir.mean():.4f}")
        
        uncertainty_factor = 1.0 + self.ot_alpha * (u_rgb + u_ir.t())
        final_cost = dist_mat * uncertainty_factor
        
        # Higher dustbin to avoid dropping valid but far matches
        dustbin_threshold = 2.5 
        
        T = solve_sinkhorn_pot(
            final_cost, 
            reg=self.ot_reg, 
            mass=self.ot_mass,
            dustbin_cost=dustbin_threshold 
        )
        
        T_np = T.cpu().numpy()
        rgb_ids_np = self.saved_rgb_ids.cpu().numpy()
        ir_ids_np = self.saved_ir_ids.cpu().numpy()
        
        # Strategy A: Bidirectional
        row_max_idx = np.argmax(T_np, axis=1) 
        col_max_idx = np.argmax(T_np, axis=0) 
        
        v2i_dict = OrderedDict()
        bidirectional_matches = 0
        N_rgb = T_np.shape[0]
        
        for i in range(N_rgb):
            j = row_max_idx[i]
            if col_max_idx[j] == i:
                if T_np[i, j] > 1e-5: 
                    r_id = rgb_ids_np[i]
                    i_id = ir_ids_np[j]
                    if r_id not in v2i_dict: v2i_dict[r_id] = []
                    v2i_dict[r_id].append(i_id)
                    bidirectional_matches += 1
        
        # Adaptive Fallback
        # If bidirectional matches are too few (<30% of classes), fallback
        fallback_triggered = False
        min_matches_needed = int(self.num_classes * 0.3) 
        
        if bidirectional_matches < min_matches_needed:
            print(f"[CMA WARNING] Only {bidirectional_matches} bidirectional matches. Triggering Unidirectional Fallback.")
            fallback_triggered = True
            v2i_dict = OrderedDict() 
            
            # Unidirectional: Trust RGB anchor, pick best IR
            # Filter by confidence
            row_max_vals = np.max(T_np, axis=1)
            confident_indices = np.argsort(-row_max_vals)
            
            num_to_keep = int(N_rgb * self.ot_mass)
            kept_indices = confident_indices[:num_to_keep]
            
            for i in kept_indices:
                j = row_max_idx[i]
                r_id = rgb_ids_np[i]
                i_id = ir_ids_np[j]
                if r_id not in v2i_dict: v2i_dict[r_id] = []
                v2i_dict[r_id].append(i_id)
                
        final_v2i = OrderedDict()
        final_i2v = OrderedDict()
        
        matched_classes = 0
        for r_id, candidates in v2i_dict.items():
            c = Counter(candidates)
            best_ir_id, count = c.most_common(1)[0]
            final_v2i[r_id] = best_ir_id
            final_i2v[best_ir_id] = r_id
            matched_classes += 1
            
        print(f"UA-POT Final: {matched_classes} classes matched (Fallback={fallback_triggered}).")
        return final_v2i, final_i2v

    def extract(self, args, model, dataset):
        model.set_eval()
        rgb_loader, ir_loader = dataset.get_normal_loader() 
        with torch.no_grad():
            rgb_f, rgb_l, rgb_cls = self._extract_one_modal(model, rgb_loader, 'rgb')
            ir_f, ir_l, ir_cls = self._extract_one_modal(model, ir_loader, 'ir')
        self.save(rgb_cls, ir_cls, rgb_l, ir_l, rgb_f, ir_f)
        
    def _extract_one_modal(self, model, loader, modal):
        saved_f, saved_l, saved_c = [], [], []
        for imgs_list, infos in loader:
            labels = infos[:, 1].to(model.device)
            imgs = imgs_list[0] if isinstance(imgs_list, list) else imgs_list
            imgs = imgs.to(model.device)
            _, bn_features = model.model(imgs)
            
            if modal == 'rgb':
                cls, _ = model.classifier1(bn_features)
            else:
                cls, _ = model.classifier2(bn_features)
            
            saved_f.append(bn_features)
            saved_l.append(labels)
            saved_c.append(cls)
        return torch.cat(saved_f), torch.cat(saved_l), torch.cat(saved_c)
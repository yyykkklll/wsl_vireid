import os
import sys
import random
import numpy as np
import torch
import time

def save_checkpoint(args, model, epoch):
    if args.dataset == 'regdb':
        path = '../saved_pretrain_{}_{}_{}_{}/'.format(args.dataset,args.arch,args.trial,args.save_path)
    else:
        path = '../saved_pretrain_{}_{}/'.format(args.dataset,args.arch)
    makedir(path)
    all_state_dict = {'backbone': model.model.state_dict(),
                    'classifier1': model.classifier1.state_dict(), 
                    'classifier2': model.classifier2.state_dict(),
                    'classifier3': model.classifier3.state_dict()}
    torch.save(all_state_dict,path+'model_{}.pth'.format(epoch))

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('make dir {} successful!'.format(path))

def time_now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().to(img.device)
    img_flip = img.index_select(3,inv_idx)
    return img_flip

@torch.no_grad()
def infoEntropy(input):
    input = torch.nn.functional.softmax(input,dim=1)
    output = -torch.mean(input*torch.log2(input + 1e-12))
    return output

class Logger:
    def __init__(self, log_file):
        self.console = sys.stdout
        self.log_file = log_file

    def __call__(self, *args):
        val = str(args[0])
        for x in args[1:]:
            val = '%s %s' % (val, str(x))
        self.console.write(val + '\n')
        with open(self.log_file, 'a') as f:
            f.write(val + '\n')

    def clear(self):
        with open(self.log_file, 'w') as f:
            pass

class MultiItemAverageMeter:
    def __init__(self):
        self.content = {}

    def update(self, val):
        for key in list(val.keys()):
            value = val[key]
            if key not in list(self.content.keys()):
                self.content[key] = {'avg': value, 'sum': value, 'count': 1.0}
            else:
                self.content[key]['sum'] += value
                self.content[key]['count'] += 1.0
                self.content[key]['avg'] = self.content[key]['sum'] / self.content[key]['count']

    def get_val(self):
        keys = list(self.content.keys())
        values = []
        for key in keys:
            try:
                values.append(self.content[key]['avg'].data.cpu().numpy())
            except:
                values.append(self.content[key]['avg'])
        return keys, values

    def get_str(self):
        result = ''
        keys, values = self.get_val()

        for i,(key, value) in enumerate(zip(keys, values)):
            result += key
            result += ': '
            result += str(value)
            result += ';  '
            if i%2:
                result += '\n'

        return result

def pha_unwrapping(x):
    fft_x = torch.fft.fft2(x.clone(), dim=(-2, -1))
    fft_x = torch.stack((fft_x.real, fft_x.imag), dim=-1)
    pha_x = torch.atan2(fft_x[:, :, :, :, 1], fft_x[:, :, :, :, 0])
    fft_clone = torch.zeros(fft_x.size(), dtype=torch.float).cuda()
    fft_clone[:, :, :, :, 0] = torch.cos(pha_x.clone())
    fft_clone[:, :, :, :, 1] = torch.sin(pha_x.clone())
    re_fft = torch.view_as_complex(fft_clone)
    re_x = torch.fft.ifft2(re_fft, dim=(-2, -1)).real
    return re_x.to(x.device)

# --- REPAIRED FUNCTION ---
def compute_uncertainty(logits):
    """
    Compute Epistemic Uncertainty using Absolute Normalized Entropy.
    Returns: (N, 1) tensor in range [0, 1] relative to max possible entropy.
    """
    # 1. Softmax
    probs = torch.softmax(logits, dim=1)
    
    # 2. Raw Entropy
    # entropy: (N, 1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1, keepdim=True)
    
    # 3. Absolute Normalization
    # Max possible entropy is log(N_classes)
    num_classes = logits.shape[1]
    max_entropy = np.log(num_classes)
    
    # Divide by max_entropy to get [0, 1] scale
    # This prevents 'Min-Max' scaling from exaggerating small differences in a confident batch
    # and prevents 'Mean=0.68' unless the model is truly confused.
    uncertainty = entropy / (max_entropy + 1e-12)
    
    return uncertainty

def solve_sinkhorn_pot(dist, reg=0.05, num_iters=20, mass=0.8, dustbin_cost=2.0):
    N, M = dist.shape
    device = dist.device
    
    C_aug = torch.zeros((N + 1, M + 1), device=device)
    C_aug[:N, :M] = dist
    C_aug[:N, M] = dustbin_cost  
    C_aug[N, :M] = dustbin_cost  
    C_aug[N, M] = 0.0            
    
    K = torch.exp(-C_aug / reg)
    
    a = torch.empty(N + 1, device=device).fill_(mass / N)
    a[N] = 1.0 - mass
    
    b = torch.empty(M + 1, device=device).fill_(mass / M)
    b[M] = 1.0 - mass
    
    u = torch.ones(N + 1, device=device) / (N + 1)
    v = torch.ones(M + 1, device=device) / (M + 1)
    
    for _ in range(num_iters):
        v = b / (torch.matmul(K.t(), u) + 1e-16)
        u = a / (torch.matmul(K, v) + 1e-16)
        
    T_aug = u.unsqueeze(1) * K * v.unsqueeze(0)
    T = T_aug[:N, :M]
    
    return T
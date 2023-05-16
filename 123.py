
import random


from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os 

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
seed = 10
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print('ok')


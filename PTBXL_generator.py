import wfdb
import numpy as np
import os
import pandas as pd
from torch.utils import data

__all__ = ['PTBXL_Dataset']

def read_PTBLX_dat(
    file_name: str,
    channels_names: list = ['i'],
    reference: str = 'age',
    ) -> tuple:
    
    chmap = {
        'i':0,'ii':1, 'iii':2, 'avr':3, 'avl':4, 'avf':5, 'V1':6, 'V2':7,
        'V3':8, 'V4':9, 'V5':10, 'V6':11
    }
    
    channels_num = [chmap[ch] for ch in channels_names]
    signals, _ = wfdb.rdsamp(file_name, channels=channels_num)
    
    return signals

class PTBXL_Dataset(data.Dataset):
    def __init__(
        self,
        path: str,
        channels: list,
        ref: str = 'sex',
        sampling_f: int = 100,
        transform = None,
    ):
        self.path = path
        self.data_key = pd.read_csv(self.path + 'ptbxl_database.csv', index_col='ecg_id')
        self.channels = channels
        self.sampling_f = sampling_f
        self.transform = transform
        self.ref = ref
        if sampling_f == 100:
            self.files = [f for f in self.data_key.filename_lr]
        else:
            self.files = [f for f in self.data_key.filename_hr]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        file = os.path.join(self.path, self.files[idx])
        record = read_PTBLX_dat(file, self.channels)
        
        marks = self.data_key.loc[self.data_key.filename_lr == self.files[idx]][self.ref].values[0]
        
        mask = (marks == -1)
        
        if self.transform:
            record, marks = self.transform(
                record,
                marks,
                sampling_f=self.sampling_f,
                fname=file,
            )
            try:
                marks[mask] = -1
            except:
                print('No marks in file')
        
        return record, marks, file
    
'''
#============== TEST SEQUENCE ==============
import matplotlib.pyplot as plt
dat_dir = 'C:/ptbxl/'
channels = ['i','ii','V1','V2','V3','V4','V5','V6']


eval_dataset = PTBXL_Dataset(
    path=dat_dir,
    channels=channels,
    sampling_f = 100,
    transform = None,
    ref = 'sex'
    )


eval_generator = data.DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)
 
for x, marks, fpaths in eval_generator:
     
    plt.plot(x[0,:,0])
    plt.show()
    print(marks)
    '''
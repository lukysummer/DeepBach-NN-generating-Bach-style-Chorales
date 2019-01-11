from abc import ABC, abstractmethod
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from process_data import PreprocessData

class LoadData():
    
    def __init__(self, cache_dir):
        self._tensor_dataset = None
        self.cache_dir = cache_dir
     
        
        
    def tensor_dataset_filepath(self, dataset_name):
        ''' Creates Cache Filepath of Tensor Dataset '''
        datasets_cache_dir = os.path.join(self.cache_dir, 'tensor_datasets')
        
        if not os.path.exists(datasets_cache_dir):
            os.mkdir(datasets_cache_dir)
        
        file_path = os.path.join(datasets_cache_dir, dataset_name)
            
        return file_path
   
    
    
    def tensor_dataset_cached(self):
        return os.path.exists(self.tensor_dataset_filepath)
            


    def tensor_dataset(self):
        '''
        Loads or Computes TensorDataset
        :return : TensorDataset
        '''
        if self._tensor_dataset is None:
            
            if self.tensor_dataset_cached():
                print("Loading Cached Tensor Dataset...")
                self._tensor_dataset = torch.load(self.tensor_dataset_filepath)
            else:
                print("Creating Tensor Dataset, since it is not cached...")
                
                self._tensor_dataset = PreprocessData.make_tensor_dataset()
                torch.save(self._tensor_dataset, self.tensor_dataset_filepath)
                
                print(f'Tensor Dataset saved in {self.tensor_dataset_filepath}')
                
        return self._tensor_dataset



    def filepath(self, file_name):
        
        tensor_datasets_cache_dir = os.path.join(self.cache_dir, 'datasets')
        
        if not os.path.exists(tensor_datasets_cache_dir):
            os.mkdir(tensor_datasets_cache_dir)
        
        file_path = os.path.join(self.cache_dir, 'datasets', file_name)
            
        return file_path
    
    
    
    def data_loaders(self,
                     batch_size,
                     split = (0.85, 0.10)):
        
        assert sum(split) < 1
        
        dataset = self.tensor_dataset
        n_samples = len(dataset)
        
        train_ratio, val_ratio = split
        train_end = int(train_ratio*n_samples)
        valid_end = int(sum(split)*n_samples)
        
        train_set = TensorDataset(*dataset[: train_end])
        valid_set = TensorDataset(*dataset[train_end : valid_end])
        test_set = TensorDataset(*dataset[valid_end:])
        
        trainloader = DataLoader(train_set,
                                 batch_size = batch_size,
                                 shuffle = True,
                                 num_workers = 4,
                                 pin_memory = True,
                                 drop_last = True) # drop the last incomplete batch
        
        validloader = DataLoader(valid_set,
                                 batch_size = batch_size,
                                 shuffle = False,
                                 num_workers = 0,
                                 pin_memory = False,
                                 drop_last = True)
        
        testloader = DataLoader(test_set,
                                batch_size = batch_size,
                                shuffle = False,
                                num_workers = 0,
                                pin_memory = False,
                                drop_last = True)
        
        return trainloader, validloader, testloader
        
        
        
        
        
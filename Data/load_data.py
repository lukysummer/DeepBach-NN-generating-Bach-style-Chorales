import os
import torch
from torch.utils.data import TensorDataset, DataLoader

from music21 import corpus

from process_data import PreprocessData
from metadata import TickMetadata, FermataMetadata, KeyMetadata
from helpers import ShortChoraleIteratorGen


all_datasets = {'bach_chorales': {'dataset_class_name': PreprocessData,
                                  'corpus_iterator': corpus.chorales.Iterator},
                'bach_chorales_test': {'dataset_class_name': PreprocessData,
                                       'corpus_iterator': ShortChoraleIteratorGen()}}

class LoadData():
    
    def __init__(self):
        
        self._tensor_dataset = None
        self.package_dir = os.path.dirname(os.path.realpath("__file__"))
        self.cache_dir = os.path.join(self.package_dir, 'dataset_cache')
        
        # Create cache dir if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
     
        
        
    def tensor_dataset_filepath(self, dataset_name):
        ''' Creates Cache Filepath of Tensor Dataset '''
        datasets_cache_dir = os.path.join(self.cache_dir, 'tensor_datasets')
        
        if not os.path.exists(datasets_cache_dir):
            os.mkdir(datasets_cache_dir)
        
        file_path = os.path.join(datasets_cache_dir, dataset_name)
            
        return file_path
   
    
    
    def tensor_dataset_cached(self):
        return os.path.exists(self.tensor_dataset_filepath)
            


    def tensor_dataset(self, make_tensor_function):
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
                
                self._tensor_dataset = make_tensor_function()
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

    
    
    def load_if_exists_or_initialize_and_save(self,
                                              dataset_name,
                                              dataset_class_name,
                                              corpus_iterator,
                                              **kwargs):
        
        kwargs.update({'dataset_name': dataset_name, 
                       'corpus_iterator': corpus_iterator,
                       'cache_dir': self.cache_dir})
    
        data_processor = dataset_class_name(**kwargs)
        
        if os.path.exists(self.filepath):
            print(f'Loading dataset from {self.filepath}...')
            dataset = torch.load(self.filepath)
            print(f'(the corresponding TensorDataset is NOT loaded)')
            
        else:
            print(f'Creating dataset... (both tensor dataset and parameters)')
            ###### Initialize & force computation of "tensor_dataset" ######
            # 1. Remove the cached data if it exists
            if os.path.exists(self.tensor_dataset_filepath):
                os.remove(self.tensor_dataset_filepath)
                
            # 2.Recompute dataset parameters and tensor_dataset
            # (this saves the tensor_dataset in self.tensor_dataset_filepath)
            tensor_dataset = self.tensor_dataset(data_processor.make_tensor_dataset())
            
            # 3. Save all dataset parameters EXCEPT the tensor dataset (stored elsewhere)
            self.tensor_dataset = None
            torch.save(dataset, self.filepath)
            
            print(f'Dataset saved in {dataset.filepath}')
            self.tensor_dataset = tensor_dataset
            
        return dataset
    
    
    
    def get_dataset(self, dataset_name:str, **dataset_kwargs) -> PreprocessData:
    
        if dataset_name == "bach_chorales":
            
            dataset = self.load_if_exists_or_initialize_and_save(
                                     dataset_name = dataset_name,
                                     dataset_class_name = "PreprocessData",
                                     corpus_iterator = corpus.chorales.Iterator,
                                     **dataset_kwargs)
            
            return dataset
    
        elif dataset_name == "bach_chorales_test":
            
            dataset = self.load_if_exists_or_initialize_and_save(
                                     dataset_name = dataset_name,
                                     dataset_class_name = "PreprocessData",
                                     corpus_iterator = ShortChoraleIteratorGen(),
                                     **dataset_kwargs)
            
            return dataset
        
        else:
            print(f'[ERROR] Dataset with dataset_name {dataset_name} is not registered!')
            raise ValueError
            
            
            
if __name__ == "__main__":
    # Usage Example
    
    dataset_loader = LoadData()

    subdivision = 4
    metadatas = [TickMetadata(subdivision = subdivision),
                 FermataMetadata(),
                 KeyMetadata()]
    
    dataset = dataset_loader.get_dataset(name = "bach_chorale_test",
                                         voice_ids = [0, 1, 2, 3],
                                         metadatas = metadatas,
                                         sequence_size = 8,
                                         subdivision = subdivision)
    
    (trainloader, validloader, testloader) = dataset_loader.data_loaders( 
                                                batch_size=128, split=(0.85, 0.10))
            
    print('Number of Train Batches: ', len(trainloader))
    print('Number of Valid Batches: ', len(validloader))
    print('Number of Test Batches: ', len(testloader))
                
        
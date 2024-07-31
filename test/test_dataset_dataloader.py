
from os import path
import unittest

from torch.utils.data import DataLoader

from titlequill.dataset import OAGKXDataset
from titlequill.utils.io_ import load_json
from titlequill.utils.logger import Logger

LOCAL_SETTINGS_FP = path.abspath(path.join(__file__, '..', '..', 'local_settings.json'))
print(LOCAL_SETTINGS_FP)

class DatasetLoaderTest(unittest.TestCase):
    
    DATASET_LEN = 14451776
    BATCH_SIZE  = 4
    
    def setUp(self) -> None: 
        
        self._logger = Logger()
        self._dataset_dir = load_json(LOCAL_SETTINGS_FP)['dataset_dir']
        
    
    def test_dataset(self):
        
        dataset = OAGKXDataset(data_dir=self._dataset_dir, logger=self._logger)
        
        self.assertEqual(len(dataset), self.DATASET_LEN)
    
    def test_loader(self):
        
        dataset = OAGKXDataset(data_dir=self._dataset_dir, logger=self._logger)
        loader  = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        
        batch = next(iter(loader))
        
        self.assertTrue(type(batch), dict)
        self.assertEqual(len(batch), 3)
        self.assertTrue('title'    in batch)
        self.assertTrue('keywords' in batch)
        self.assertTrue('abstract' in batch)


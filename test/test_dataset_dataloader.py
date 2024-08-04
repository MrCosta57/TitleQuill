
import unittest


from script.utils.settings import DATASET_DIR, DATASET_TSV_FILE
from titlequill.dataset import OAGKXItem, OAGKXRawDataset, OAGKXTSVDataset
from titlequill.utils.logger import Logger

class OAGKXRawDatasetTest(unittest.TestCase):
    
    DATASET_LEN = 14451776
    
    
    def setUp(self) -> None: 
        
        self._dataset = OAGKXRawDataset(dataset_dir=DATASET_DIR, logger=Logger())
    
    def test_dataset(self):
        
        self.assertTrue(isinstance(self._dataset[0], OAGKXItem))
        self.assertEqual(len(self._dataset), self.DATASET_LEN)
    
    def test_dataloader(self):
        
        loader  = self._dataset.get_dataloader()
        
        batch = next(iter(loader))
        
        self.assertTrue(type(batch), dict)
        self.assertEqual(len(batch), 3)
        self.assertTrue([key in batch for key in OAGKXItem.KEYS])
    
    def test_filtering(self):

        self._dataset.apply_filter(lambda item: False)
        self.assertEqual(len(self._dataset), 0)

class OAGKXTSVDatasetTest(unittest.TestCase):
    
    def setUp(self) -> None: 
        
        self._logger = Logger()
        self._dataset = OAGKXTSVDataset(file_path=DATASET_TSV_FILE)
        
    def test_dataset(self):
        
        self.assertTrue(isinstance(self._dataset[0], OAGKXItem))
    
    def test_dataloader(self):
        
        loader  = self._dataset.get_dataloader()
        
        batch = next(iter(loader))
        
        self.assertTrue(type(batch), dict)
        self.assertEqual(len(batch), 3)
        self.assertTrue([key in batch for key in OAGKXItem.KEYS])
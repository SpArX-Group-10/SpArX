from module1 import import_dataset
import unittest

class Module1Test(unittest.TestCase):
    
    def test_import_dataset(self):
        filepath = "test_data.csv"
        data_entries, labels = import_dataset(filepath)
        print(data_entries)
        print(labels)
        self.assertEqual(data_entries.shape, (3, 2))
        self.assertEqual(labels.shape, (3,))

if __name__ == '__main__':
    Module1Test().test_import_dataset()

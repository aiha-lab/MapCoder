from Datasets.Dataset import Dataset
from Datasets.MBPPDataset import MBPPDataset
from Datasets.APPSDataset import APPSDataset
from Datasets.XCodeDataset import XCodeDataset
from Datasets.HumanEvalDataset import HumanDataset
from Datasets.CodeContestDataset import CodeContestDataset


class DatasetFactory:
    @staticmethod
    def get_dataset_class(dataset_name):
        if dataset_name == "APPS":
            return APPSDataset
        elif dataset_name == "MBPP":
            return MBPPDataset
        elif dataset_name == "XCode":
            return XCodeDataset
        elif dataset_name == "HumanEval":
            return HumanDataset
        elif dataset_name == "Human":
            return HumanDataset
        elif dataset_name == "CC":
            return CodeContestDataset
        else:
            raise Exception(f"Unknown dataset name {dataset_name}")

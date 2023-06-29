import torch as t
from torch.utils import data
from FraGAT.ChemUtils import *
import rdkit.Chem as Chem
import re
import random
from FraGAT.Featurizer import *
from FraGAT.Transformer import *
from FraGAT.Splitter import *
from FraGAT.Checker import *

class FileLoader(object):
    def __init__(self, file_path):
        super(FileLoader, self).__init__()
        self.path = file_path

    def load(self):
        with open(self.path, 'r') as f:
            raw_data = f.readlines()
        return raw_data


class BasicFileParser(object):
    def __init__(self):
        super(BasicFileParser, self).__init__()

    def _parse_line(self, line):
        raise NotImplementedError(
            "Line parser not implemented.")

    def parse_file(self, raw_data):

        Dataset = []
        for line in raw_data:
            data = self._parse_line(line)
            Dataset.append(data)
        return Dataset


class ESOLFileParser(BasicFileParser):
    def __init__(self):
        super(ESOLFileParser, self).__init__()

    def _parse_line(self, line):
        data = re.split(',',line)
        SMILES = data[0]
        Value = data[1]
        Value = re.split('\n', Value)[0]
        return {'SMILES': SMILES, 'Value': Value}

class MolDatasetEval(data.Dataset):
    def __init__(self, dataset, opt):
        super(MolDatasetEval, self).__init__()
        self.dataset = dataset
        self.Frag = opt.args['Frag']
        self.opt = opt
        self.FeaturizerList = {
            'AttentiveFP': AttentiveFPFeaturizer(
                atom_feature_size=opt.args['atom_feature_size'],
                bond_feature_size=opt.args['bond_feature_size'],
                max_degree = 5,
                max_frag = 2,
                mode='EVAL'
            )

        }
        self.featurizer = self.FeaturizerList[opt.args['Feature']]

        # if use methods in AttentiveFP to construct dataset, some more works should be down here.
        if opt.args['Feature'] == 'AttentiveFP':
            print("Using Attentive FP. Dataset is being checked.")
            self.checker = AttentiveFPChecker(max_atom_num=102, max_degree=5)
            self.dataset = self.checker.check(self.dataset)       # screen invalid molecules
            print("Prefeaturizing molecules......")
            self.featurizer.GetPad(self.dataset)
            self.prefeaturized_dataset = self.featurizer.prefeaturize(self.dataset)
            print("Prefeaturization complete.")

    def __getitem__(self, index):
        if self.featurizer.__class__ == AttentiveFPFeaturizer:
            value = self.dataset[index]["Value"]
            smiles = self.dataset[index]["SMILES"]
            mol = Chem.MolFromSmiles(smiles)
            #print("Single bonds num: ", len(GetSingleBonds(mol)))
            data, label = self.featurizer.featurizenew(self.prefeaturized_dataset, index, mol, value, self.Frag, self.opt)
        else:
            item = self.dataset[index]
            data, label = self.featurizer.featurize(item)
        return data, label

    def __len__(self):
        return len(self.dataset)

class MolDatasetTrain(data.Dataset):
    def __init__(self, dataset, opt):
        super(MolDatasetTrain, self).__init__()
        self.dataset = dataset
        self.opt = opt
        self.Frag = self.opt.args['Frag']
        self.FeaturizerList = {
            'AttentiveFP': AttentiveFPFeaturizer(
                atom_feature_size=opt.args['atom_feature_size'],
                bond_feature_size=opt.args['bond_feature_size'],
                max_degree = 5,
                max_frag = 2,
                mode='TRAIN'
            )

        }
        self.featurizer = self.FeaturizerList[opt.args['Feature']]
        if 'max_atom_num' in self.opt.args:
            self.max_atom_num = self.opt.args['max_atom_num']
        else:
            self.max_atom_num = 102

        if opt.args['Feature'] == 'AttentiveFP':
            print("Using Attentive FP. Dataset is being checked.")
            self.checker = AttentiveFPChecker(max_atom_num=self.max_atom_num, max_degree=5)
            self.dataset = self.checker.check(self.dataset)       # screen invalid molecules
            print("Prefeaturizing molecules......")
            self.featurizer.GetPad(self.dataset)
            self.prefeaturized_dataset = self.featurizer.prefeaturize(self.dataset)
            print("Prefeaturization complete.")

    def __getitem__(self, index):
        if self.featurizer.__class__ == AttentiveFPFeaturizer:
            value = self.dataset[index]["Value"]
            smiles = self.dataset[index]["SMILES"]
            mol = Chem.MolFromSmiles(smiles)
            data, label = self.featurizer.featurizenew(self.prefeaturized_dataset, index, mol, value, self.Frag, self.opt)
        else:
            item = self.dataset[index]
            data, label = self.featurizer.featurize(item)
        return data, label

    def __len__(self):
        return len(self.dataset)


class MolDatasetCreator(object):

    def __init__(self, opt):
        super(MolDatasetCreator, self).__init__()

        self.FileParserList = {
            'ESOL': ESOLFileParser()
        }
        self.SplitterList = {
            'Random': RandomSplitter(),
            'Scaffold': ScaffoldSplitter(),
        }
        self.TransformerList = {
            'Augmentation': BinaryClassificationAugmentationTransformer()
        }

        self.opt = opt

    def CreateDatasets(self):
        file_path = self.opt.args['DataPath']
        print("Loading data file...")
        fileloader = FileLoader(file_path)
        raw_data = fileloader.load()

        print("Parsing lines...")
        parser = self.FileParserList[self.opt.args['ExpName']]
        raw_dataset = parser.parse_file(raw_data)
        # raw_dataset is a list in type of : {'SMILES': , 'Value': }
        print("Dataset is parsed. Original size is ", len(raw_dataset))

        if self.opt.args['ClassNum'] == 2:         # only binary classification tasks needs to calculate weights.
            if self.opt.args['Weight']:
                weights = self.CalculateWeight(raw_dataset)
            else:
                weights = None
        else:
            weights = None


        if self.opt.args['Splitter']:
            if (self.opt.args['Splitter'] == 'Scaffold') ):
                checker = ScaffoldSplitterChecker()
                raw_dataset = checker.check(raw_dataset)
                # if use scaffold splitter, the invalid smiles should be discarded from raw_dataset first.

            splitter = self.SplitterList[self.opt.args['Splitter']]
            print("Splitting dataset...")
            sets = splitter.split(raw_dataset, self.opt)
            if len(sets) == 2:
                trainset, validset = sets
                print("Dataset is splitted into trainset: ", len(trainset), " and validset: ", len(validset))
            if len(sets) == 3:
                trainset, validset, testset = sets
                print("Dataset is splitted into trainset: ", len(trainset), ", validset: ", len(validset), " and testset: ", len(testset))
        else:
            trainset = raw_dataset
            sets = (trainset)

        # if the weight is used, then the augmentation should not be used.
        if self.opt.args['ClassNum'] == 2:
            if not self.opt.args['Weight']:
                if self.opt.args['Augmentation']:
                    transformer = self.TransformerList['Augmentation']
                    print("Augmentating...")
                    trainset = transformer.transform(trainset)
                    print("Trainset is enlarged to size: ", len(trainset))
                    if self.opt.args['ValidBalance']:
                        validset = transformer.transform(validset)
                        print("Validset is enlarged to size: ", len(validset))
                    if self.opt.args['TestBalance']:
                        testset = transformer.transform(testset)
                        print("Testset is enlarged to size: ", len(testset))


        Trainset = MolDatasetTrain(trainset, self.opt)
        if len(sets) == 2:
            Validset = MolDatasetEval(validset, self.opt)
            return (Trainset, Validset), weights
        elif len(sets) == 3:
            Validset = MolDatasetEval(validset, self.opt)
            Testset = MolDatasetEval(testset, self.opt)
            return (Trainset, Validset, Testset), weights
        else:
            return (Trainset), weights

    def CalculateWeight(self, dataset):
        weights = []
        task_num = self.opt.args['TaskNum']
        for i in range(task_num):
            pos_count = 0
            neg_count = 0
            for item in dataset:
                value = item['Value'][i]
                if value == '0':
                    neg_count += 1
                elif value == '1':
                    pos_count += 1
            pos_weight = (pos_count + neg_count) / pos_count
            neg_weight = (pos_count + neg_count) / neg_count
            weights.append([neg_weight, pos_weight])
        return weights

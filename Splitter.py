import random
import numpy as np
from FraGAT.ChemUtils import *

class BasicSplitter(object):
    def __init__(self):
        super(BasicSplitter, self).__init__()
    def split(self, dataset, opt):
        raise NotImplementedError(
            "Dataset Splitter not implemented.")

class RandomSplitter(BasicSplitter):
    def __init__(self):
        super(RandomSplitter, self).__init__()

    def CheckClass(self, dataset, tasknum):
        c0cnt = np.zeros(tasknum)
        c1cnt = np.zeros(tasknum)
        for data in dataset:
            value = data['Value']
            assert tasknum == len(value)
            for task in range(tasknum):
                if value[task] == '0':
                    c0cnt[task]+=1
                elif value[task] == '1':
                    c1cnt[task]+=1
        if 0 in c0cnt:
            print("Invalid splitting.")
            return False
        elif 0 in c1cnt:
            print("Invalid splitting.")
            return False
        else:
            return True

    def split(self, dataset, opt):
        
        #SplitRate:8:1:1,7:2:1,6:3:1.5:4:1
        
        rate = opt.args['SplitRate']
        validseed = opt.args['SplitValidSeed']
        testseed = opt.args['SplitTestSeed']
        total_num = len(dataset)
        if len(rate) == 1:
            train_num = int(total_num * rate[0])
            valid_num = total_num - train_num
            endflag = 0
            while not endflag:
                random.seed(validseed)
                random.shuffle(dataset)
                set1 = dataset[:train_num]
                set2 = dataset[train_num:]

                assert len(set1) == train_num
                assert len(set2) == valid_num
                endflag = 1
            return (set1, set2)

        if len(rate) == 2:
            train_num = int(total_num * rate[0])
            valid_num = int(total_num * rate[1])
            test_num = total_num - train_num - valid_num
            endflag = 0
            while not endflag:
                random.seed(testseed)
                random.shuffle(dataset)
                set3 = dataset[(train_num + valid_num):]
                endflag = 1

            set_remain = dataset[:(train_num + valid_num)]
            endflag = 0
            while not endflag:
                random.seed(validseed)
                random.shuffle(set_remain)
                set1 = set_remain[:train_num]
                set2 = set_remain[train_num:]
                endflag = 1

                assert len(set1) == train_num
                assert len(set2) == valid_num
                assert len(set3) == test_num

            return (set1,set2,set3)

class ScaffoldRandomSplitter(BasicSplitter):
    def __init__(self):
        super(ScaffoldRandomSplitter, self).__init__()

    def generate_scaffold(self, smiles, include_chirality = False):
        generator = ScaffoldGenerator(include_chirality = include_chirality)
        scaffold = generator.get_scaffold(smiles)
        return scaffold

    def id2data(self, dataset, ids):
        new_dataset = []
        for id in ids:
            data = dataset[id]
            new_dataset.append(data)
        return new_dataset

    def split(self, dataset, opt):
        rate = opt.args['SplitRate']
        validseed = opt.args['SplitValidSeed']
        testseed = opt.args['SplitTestSeed']
        total_num = len(dataset)
        scaffolds = {}

        # extract scaffolds.
        for id, data in enumerate(dataset):
            smiles = data['SMILES']
            scaffold = self.generate_scaffold(smiles)

            if scaffold not in scaffolds:
                scaffolds[scaffold] = [id]
            else:
                scaffolds[scaffold].append(id)

        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}

        if len(rate) == 1:
            assert rate[0] < 1
            train_num = int(total_num * rate[0])
            valid_num = total_num - train_num
        elif len(rate) == 2:
            assert rate[0]+rate[1] < 1
            train_num = int(total_num * rate[0])
            valid_num = int(total_num * rate[1])
            test_num = total_num - train_num - valid_num
        else:
            print("Wrong splitting rate")
            raise RuntimeError

        tasknum = opt.args['TaskNum']  #1
        classnum = opt.args['ClassNum']  # for regression task, classnum is sest to be 1


        scaffold_keys = scaffolds.keys()  # sample scaffolds from scaffold_keys
        if len(rate) == 1: 
            sample_size = int(len(scaffold_keys) * (1 - rate[0]))

            validids, _ = self.BinaryClassSample(dataset, scaffolds, sample_size, valid_num, minor_ratio, minorclass, validseed)
            validset = self.id2data(dataset, validids)
            trainids = self.excludedids(len(dataset), validids)
            trainset = self.id2data(dataset, trainids)
            return (trainset, validset)
        elif len(rate) == 2:  
            sample_size = int(len(scaffold_keys) * (1 - rate[0] - rate[1]))
            testids, chosen_scaffolds = self.BinaryClassSample(dataset, scaffolds, sample_size, test_num, minor_ratio,
                                                        minorclass, testseed)
            testset = self.id2data(dataset, testids)

            remain_scaffolds = {x: scaffolds[x] for x in scaffolds.keys() if x not in chosen_scaffolds}
            sample_size = int(len(remain_scaffolds.keys()) * rate[1])
            validids, _ = self.BinaryClassSample(dataset, remain_scaffolds, sample_size, valid_num, minor_ratio, minorclass,
                                          validseed)
            validset = self.id2data(dataset, validids)
            trainids = self.excludedids(len(dataset), validids + testids)
            trainset = self.id2data(dataset, trainids)
            return (trainset, validset, testset)

        elif classnum == 1: # regression
            scaffold_keys = scaffolds.keys() 
            if len(rate) == 1:  
                sample_size = int(len(scaffold_keys) * (1 - rate[0]))

                validids, _ = self.RegressionSample(scaffolds, sample_size, valid_num, validseed)
                validset = self.id2data(dataset, validids)
                trainids = self.excludedids(len(dataset), validids)
                trainset = self.id2data(dataset, trainids)
                return (trainset, validset)
            elif len(rate) == 2:  
                sample_size = int(len(scaffold_keys) * (1 - rate[0] - rate[1]))
                testids, chosen_scaffolds = self.RegressionSample(scaffolds, sample_size, test_num, testseed)
                testset = self.id2data(dataset, testids)
               
                remain_scaffolds = {x: scaffolds[x] for x in scaffolds.keys() if x not in chosen_scaffolds}
                sample_size = int(len(remain_scaffolds.keys()) * rate[1])
                validids, _ = self.RegressionSample(remain_scaffolds, sample_size, valid_num, validseed)
                validset = self.id2data(dataset, validids)
                trainids = self.excludedids(len(dataset), validids + testids)
                trainset = self.id2data(dataset, trainids)
                return (trainset, validset, testset)


    def RegressionSample(self, scaffolds, sample_size, optimal_count, seed):
        count = 0
        keys = scaffolds.keys()
        tried_times = 0
        error_rate = 0.1
        while (count < optimal_count * (1-error_rate)) or (count > optimal_count * (1+error_rate)):
            tried_times += 1
            if tried_times % 5000 == 0:
                print("modify error rate.")
                error_rate += 0.05
                print("modify sample scaffold number.")
                sample_size = int(sample_size * 1.1)
                print(len(list(scaffolds.keys())))
                print(sample_size)
            seed += 1
            random.seed(seed)

            chosen_scaffolds = random.sample(list(keys), sample_size)
            count = sum([len(scaffolds[scaffold]) for scaffold in chosen_scaffolds])
            index = [index for scaffold in chosen_scaffolds for index in scaffolds[scaffold]]

        print("Sample num: ", count)
        print("Available Seed: ", seed)
        print("Tried times: ", tried_times)
        return index, chosen_scaffolds

    def id2valuecount(self, dataset, ids, count_value):
        count = 0
        for id in ids:
            data = dataset[id]
            value = data['Value']
            if value == count_value:
                count += 1
        return count

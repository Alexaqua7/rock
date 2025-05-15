from dataset import CustomDataset
from torch.utils.data import DataLoader
from loss import weighted_normalized_CrossEntropyLoss, weighted_normalized_CrossEntropyLoss_custom, weighted_normalized_CrossEntropyLoss_diff_weighted
from sampling import WeightedRandomSampler, create_weighted_sampler, HardNegativeMiner, BalancedHardNegativeBatchSampler
from training_function import train, hard_negative_train, validation, hard_negative_validation
from utils import seed_everything
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import timm
import wandb
import torch
import json
import platform
import socket
import getpass
import os

MAPPER = {'internimage': 'huggingface'}


class Trainer:
    def __init__(self, CFG):
        self.CFG = {k.upper(): v for k, v in CFG.items()}
        seed_everything(self.CFG['SEED'])
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def _set_training_mode(self) -> None:
        """Determine the training mode based on configuration settings."""
        if (self.config.get('HARD_NEGATIVE_MEMORY_SIZE', 0) > 0 and self.config.get('HARD_NEGATIVE_RATIO', 0) > 0):
            self.config['TRAIN_MODE'] = 'TRAIN_MODE_HARD_NEGATIVE'
        elif self.config.get('BALANCED_BATCH', False):
            self.config['TRAIN_MODE'] = 'TRAIN_MODE_OVERSAMPLE'
        else:
            self.config['TRAIN_MODE'] = 'TRAIN_MODE_BASE'

    def init_dataset(self, mode='train'):
        all_img_list = glob.glob('./train/*/*')
        df = pd.DataFrame(columns=['img_path', 'rock_type'])
        df['img_path'] = all_img_list
        df['rock_type'] = df['img_path'].apply(lambda x : str(x).replace('\\','/').split('/')[2])

        train_data, val_data, _, _ = train_test_split(df, df['rock_type'], test_size=self.CFG['TEST_SIZE'], stratify=df['rock_type'], random_state=self.CFG['SEED'])
        le = preprocessing.LabelEncoder()
        train_data['rock_type'] = le.fit_transform(train_data['rock_type'])
        val_data['rock_type'] = le.transform(val_data['rock_type'])

        class_names = le.classes_
        num_classes = len(class_names)

        train_transform = self.CFG['TRAIN_TRANSFORM']
        test_transform = self.CFG['TEST_TRANSFORM']

        if mode == 'train':
            train_dataset = CustomDataset(train_data['img_path'].values, train_data['rock_type'].values, train_transform)
            val_dataset = CustomDataset(val_data['img_path'].values, val_data['rock_type'].values, test_transform)

            return {"train_dataset": train_dataset, "val_dataset": val_dataset, "class_names": class_names, 'num_classes': num_classes, 'train_data': train_data, 'val_data': val_data}
        elif mode == 'test':
            pass #TODO 구현해야 함
        else:
            raise ValueError('mode should be either train or test')
    
    def init_loader(self, mode='train', train_mode='BASE', dataset: dict=None):
        if dataset is None:
            raise SyntaxError('Please Pass the Dataset')
        if mode == 'train':
            train_dataset, val_dataset = dataset['train_dataset'], dataset['val_dataset']
            if train_mode == 'BASE':
                train_loader = DataLoader(train_dataset, batch_size=self.CFG['BATCH_SIZE'], shuffle=True, num_workers=self.CFG['NUM_WORKERS'], pin_memory=True, prefetch_factor=4)
                val_loader = DataLoader(val_dataset, batch_size=self.CFG['BATCH_SIZE'], shuffle=False, num_workers=self.CFG['NUM_WORKERS'], pin_memory=True, prefetch_factor=4)

                return {'train_loader': train_loader, 'val_loader': val_loader}
            
            elif train_mode == 'OVERSAMPLE':
                train_data, _ = self.get_train_data()
                sampler = create_weighted_sampler(train_data['rock_type'].values)
                train_loader = DataLoader(train_dataset, batch_size=self.CFG['BATCH_SIZE'], sampler=sampler, num_workers=self.CFG['NUM_WORKERS'], pin_memory=True, prefetch_factor=4)
                val_loader = DataLoader(val_dataset, batch_size=self.CFG['BATCH_SIZE'], shuffle=False, num_workers=self.CFG['NUM_WORKERS'], pin_memory=True, prefetch_factor=4)

                return {'train_loader': train_loader, 'val_loader': val_loader}
            
            elif train_mode == 'HARDNEGATIVESAMPLE':
                hard_negative_miner = HardNegativeMiner(dataset_size=len(train_data), memory_size=self.CFG['HARD_NEGATIVE_MEMORY_SIZE'])
                balanced_batch_sampler = BalancedHardNegativeBatchSampler(dataset_size=len(train_dataset), batch_size=self.CFG['BATCH_SIZE'], hard_negative_miner=hard_negative_miner,
                labels=dataset['train_data']['rock_type'].values,  # 클래스 레이블 전달
                num_classes=dataset['num_classes'],  # 클래스 수 전달
                hard_negative_ratio=self.CFG['HARD_NEGATIVE_RATIO']
            )
                # 균형 잡힌 배치 샘플러를 사용한 데이터로더
                train_loader = DataLoader(
                    train_dataset, 
                    batch_sampler=balanced_batch_sampler,
                    num_workers=self.CFG['NUM_WORKERS'], 
                    pin_memory=True
                )
                val_loader = DataLoader(val_dataset, batch_size=self.CFG['BATCH_SIZE'], shuffle=False, num_workers=self.CFG['NUM_WORKERS'], pin_memory=True, prefetch_factor=4)

                return {'train_loader': train_loader, 'val_loader': val_loader, 'hard_negative_miner': hard_negative_miner}
            else:
                raise SyntaxError('Not Implemented!')
        
        elif mode =='test':
            pass #TODO 구현해야 함
    
    
    def init_model(self, class_names):
        model_type = MAPPER.get(self.CFG['MODEL_NAME'])
        if model_type == 'huggingface':
            pass #TODO 구현해야 함
            
        elif model_type == None:
            model = timm.create_model(self.CFG['MODEL_NAME'], pretrained=True, num_classes=len(class_names))
        
        return model
    
    def set_experiment(self):
        trained_path = self.CFG['TRAINED_PATH']

        if trained_path == "":
            idx = len([x for x in os.listdir('./experiments') if x.startswith(self.CFG['MODEL_NAME'])])
            experiment_name = f"{self.CFG['MODEL_NAME'].replace('.','_')}_{idx+1}" # 실험이 저장될 folder 이름
        else:
            experiment_name = os.path.splitext(os.path.basename(trained_path))[0].split('-')[0]
        folder_path = os.path.join("./experiments", experiment_name)
        os.makedirs(folder_path, exist_ok=True)
        # 실험 설정 저장
        config = {'experiment': {}, 'model':{}, 'train':{}, 'validation':{}, 'split':{}, 'seed': {}}
        config['experiment']['name'] = experiment_name

        config['model']['name'] = self.CFG['MODEL_NAME']
        config['model']['IMG_size'] = self.CFG['IMG_SIZE']

        config['train']['epoch'] = self.CFG['EPOCHS']
        config['train']['lr'] = self.CFG['LEARNING_RATE']
        config['train']['train_transform'] = [str(x) for x in self.CFG['TRAIN_TRANSFORM']]
        config['train']['optimizer'] = {}
        config['train']['optimizer']['name'] = self.CFG['OPTIMIZER'].__class__.__name__
        config['train']['scheduler'] = {}
        config['train']['scheduler']['name'] = self.CFG['SCHEDULER'].__class__.__name__
        config['train']['hard_negative_ratio'] = self.CFG['HARD_NEGATIVE_RATIO']
        config['train']['hard_negative_memory_size'] = self.CFG['HARD_NEGATIVE_MEMORY_SIZE']
        config['train']['balanced_class_sampling'] = True  # 클래스 균형 샘플링 적용 여부 추가

        config['validation']['test_transform'] = [str(x) for x in self.CFG['TRAIN_TRANSFORM']]

        config['split'] = self.CFG['TEST_SIZE']

        config['seed'] = self.CFG['SEED']

        config['system'] = {
            'hostname': socket.gethostname(),
            'username': getpass.getuser(),
            'platform': platform.system(),
            'platform-release': platform.release(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
        }

        for k, v in self.CFG['OPTIMIZER'].state_dict()['param_groups'][0].items():
            if k == 'params': continue
            config['train']['optimizer'][k] = v

        for k, v in self.CFG['SCHEDULER'].state_dict().items():
            if k == 'params': continue
            config['train']['scheduler'][k] = v
            
        experiment_dir = f"./experiments/{experiment_name}"
        os.makedirs(experiment_dir, exist_ok=True)
        config_path = os.path.join(experiment_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        return {'experiment_dir': experiment_dir, 'folder_path': folder_path}

    def train(self):
        _dataset = self.init_dataset('train')

        _loader = self.init_loader('train', self.CFG['TRAIN_MODE'], _dataset)
        train_loader, val_loader = _loader['train_loader'], _loader['val_loader']

        optimizer = self.CFG['OPTIMIZER']
        scheduler = self.CFG['SCHEDULER']

        model = self.init_model(_dataset['class_names'])
        _exp_settings = self.set_experiment()
        # wandb config 업데이트
        wandb.config.update({
            "optimizer": optimizer.__class__.__name__,
            "scheduler": scheduler.__class__.__name__,
            "model": self.CFG['model_name'],
            "hard_negative_ratio": self.CFG['HARD_NEGATIVE_RATIO'],
            "hard_negative_memory_size": self.CFG['HARD_NEGATIVE_MEMORY_SIZE'],
            "balanced_class_sampling": True if self.CFG['TRAIN_MODE'] == 'OVERSAMPLE' or self.CFG['HARDNEGATIVESAMPLE'] else False# 클래스 균형 샘플링 적용 여부 추가
        })
        if self.CFG['TRAIN_MODE'] == 'BASE':
            infer_model = train(model, optimizer, train_loader, val_loader, scheduler, self.device, _dataset['class_names'], experiment_name=_exp_settings['experiment_name'], folder_path = _exp_settings['folder_path'], class_counts=_dataset['class_counts'], accumulation_steps=self.CFG['ACCUMULATION_STEPS'])
        
        elif self.CFG['TRAIN_MODE'] == 'OVERSAMPLE':
            infer_model = train(model, optimizer, train_loader, val_loader, scheduler, self.device, _dataset['class_names'], experiment_name=_exp_settings['experiment_name'], folder_path=_exp_settings['folder_path'], class_counts=_dataset['class_counts'], accumulation_steps=self.CFG['ACCUMULATION_STEPS'])
        
        elif self.CFG['TRAIN_MODE'] == 'HARDNEGATIVESAMPLE':
            infer_model = train(model, optimizer, train_loader, val_loader, scheduler, self.device, _dataset['class_names'], 
                           hard_negative_miner=_loader['hard_negative_sample'], experiment_name=_exp_settings['experiment_name'], folder_path=_exp_settings['folder_path'], accumulation_steps=self.CFG['ACCUMULATION_STEPS'])
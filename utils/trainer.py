import warnings
from utils.dataset import CustomDataset
from torch.utils.data import DataLoader
from utils.loss import weighted_normalized_CrossEntropyLoss, weighted_normalized_CrossEntropyLoss_custom, weighted_normalized_CrossEntropyLoss_diff_weighted
from utils.sampling import create_weighted_sampler, HardNegativeMiner, create_train_loader_with_accumulation, ProgressiveScheduler
from utils.training_function import train, hard_negative_train, progressive_hard_negative_train
from utils.utils import seed_everything
import glob
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
import timm
import wandb
import torch
import json
import platform
import socket
import getpass
import os
from typing import Dict, Any, Optional, Tuple, List
from collections import Counter
from tqdm import tqdm
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

# Model source mapping
MODEL_SOURCE_MAPPER = {
    'internimage': 'huggingface',
    # Add more mappings as needed
}

# Training modes
TRAIN_MODE_BASE = 'BASE'
TRAIN_MODE_OVERSAMPLE = 'OVERSAMPLE'
TRAIN_MODE_HARD_NEGATIVE = 'HARD_NEGATIVE_SAMPLE'
TRAIN_MODE_PROGRESSIVE_HARD_NEGATIVE = 'PROGRESSIVE_HARD_NEGATIVE_SAMPLE'


class Trainer:
    """
    A comprehensive trainer class for managing the machine learning training pipeline.
    
    This class handles dataset preparation, model initialization, training configurations,
    and experiment tracking for various training modes including standard training,
    oversampling, and hard negative sampling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Trainer with configuration settings.
        
        Args:
            config: Dictionary containing all configuration parameters
        """
        # Ensure configuration keys are consistent (uppercase)
        self.config = {k.upper(): v for k, v in config.items()}
        
        # Set seed for reproducibility
        seed_everything(self.config['SEED'])
        
        # Determine training mode
        self._set_training_mode()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create base paths
        self.data_path = self.config.get('DATA_PATH', './train')
        self.experiment_path = self.config.get('EXPERIMENT_PATH', './experiments')
        
        # Ensure directory exists
        os.makedirs(self.experiment_path, exist_ok=True)
    
    def _set_training_mode(self) -> None:
        """Determine the training mode based on configuration settings."""

        if (self.config.get('INITIAL_RATIO'), None) and (self.config.get('FINAL_RATIO', None)) and self.config.get('HARD_NEGATIVE_MEMORY_SIZE', 0) > 0:
            self.config['TRAIN_MODE'] = TRAIN_MODE_PROGRESSIVE_HARD_NEGATIVE
        elif (self.config.get('HARD_NEGATIVE_MEMORY_SIZE', 0) > 0 and 
            self.config.get('HARD_NEGATIVE_RATIO', 0) > 0):
            self.config['TRAIN_MODE'] = TRAIN_MODE_HARD_NEGATIVE
        elif self.config.get('BALANCED_BATCH', False):
            self.config['TRAIN_MODE'] = TRAIN_MODE_OVERSAMPLE
        else:
            self.config['TRAIN_MODE'] = TRAIN_MODE_BASE
    
    def init_dataset(self, mode: str = 'train') -> Dict[str, Any]:
        """
        Initialize dataset for training or testing.
        
        Args:
            mode: 'train' or 'test'
            
        Returns:
            Dictionary containing dataset objects and related information
        
        Raises:
            ValueError: If mode is not 'train' or 'test'
        """
        if mode not in ['train', 'test']:
            raise ValueError("Mode should be either 'train' or 'test'")
        
        if mode == 'train':
            # Get image paths
            all_img_list = glob.glob(os.path.join(self.data_path, '*', '*'))
            
            # Create DataFrame
            df = pd.DataFrame(columns=['img_path', 'rock_type'])
            df['img_path'] = all_img_list
            df['rock_type'] = df['img_path'].apply(
                lambda x: str(x).replace('\\', '/').split('/')[-2]
            )

            le = preprocessing.LabelEncoder()
            df['rock_type'] = le.fit_transform(df['rock_type'])

            df, _, _, _ = train_test_split(
                    df, 
                    df['rock_type'], 
                    test_size=0.99, 
                    stratify=df['rock_type'], 
                    random_state=self.config['SEED']
                )

            # Split data into train and validation sets
            if self.config['FOLD'] > 0:
                kfold = StratifiedKFold(n_splits=self.config['FOLD'], shuffle=True, random_state=self.config['SEED'])
                kfold_data = {}
                for fold, (train_idx, val_idx) in enumerate(kfold.split(df['img_path'], df['rock_type'])):
                    kfold_data[fold] = (train_idx, val_idx)

            else:
                train_data, val_data, _, _ = train_test_split(
                    df, 
                    df['rock_type'], 
                    test_size=self.config['TEST_SIZE'], 
                    stratify=df['rock_type'], 
                    random_state=self.config['SEED']
                )
            
            # Get class information
            class_names = le.classes_
            num_classes = len(class_names)
            
            kFold_class_infos = dict()
            if self.config['FOLD'] > 0:
                for fold in range(self.config['FOLD']):
                    label_counts = Counter(df.iloc[kfold_data[fold][0]].copy().reset_index(drop=True)['rock_type'])
                    class_counts = {
                class_name: label_counts[i] for i, class_name in enumerate(le.classes_)
            }
                    kFold_class_infos[fold] = {"label_counts": label_counts, "class_counts": class_counts}
            else:
                label_counts = Counter(train_data['rock_type'])
                # le.classes_ 순서에 맞춰 클래스별 count 매핑
                class_counts = {
                class_name: label_counts[i] for i, class_name in enumerate(le.classes_)
            }
            
            # Initialize datasets with transformations
            train_transform = self.config['TRAIN_TRANSFORM']
            test_transform = self.config['TEST_TRANSFORM']
            

            if self.config['FOLD'] > 0:
                kFold_dataset = dict()
                for fold in range(self.config['FOLD']):
                    train_idx, val_idx = kfold_data[fold]
                    train_data = df.iloc[train_idx].copy().reset_index(drop=True)
                    val_data = df.iloc[val_idx].copy().reset_index(drop=True)

                    train_dataset = CustomDataset(
                    train_data['img_path'].values, 
                    train_data['rock_type'].values, 
                    train_transform
                )
                    val_dataset = CustomDataset(
                    val_data['img_path'].values, 
                    val_data['rock_type'].values, 
                    test_transform
                )
                    kFold_dataset[fold] = {
                    'train_dataset': train_dataset,
                    'val_dataset': val_dataset,
                    'class_names': class_names,
                    'num_classes': num_classes,
                    'train_data': train_data,
                    'val_data': val_data,
                    'class_counts': kFold_class_infos[fold]['class_counts'],
                    'le': le
                }

                return kFold_dataset
            
            else:
                train_dataset = CustomDataset(
                    train_data['img_path'].values, 
                    train_data['rock_type'].values, 
                    train_transform
                )
                val_dataset = CustomDataset(
                    val_data['img_path'].values, 
                    val_data['rock_type'].values, 
                    test_transform
                )
                
                return {
                    "train_dataset": train_dataset, 
                    "val_dataset": val_dataset, 
                    "class_names": class_names, 
                    'num_classes': num_classes, 
                    'train_data': train_data, 
                    'val_data': val_data,
                    'class_counts': class_counts,
                    'le': le
            }
        elif mode == 'test':
            # Load test data
            test = pd.read_csv('./test.csv')
            test_transform = self.config['TEST_TRANSFORM']
            test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
            
            return {"test_dataset": test_dataset}
    
    def init_loader(self, mode: str = 'train', dataset: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Initialize data loaders based on the training mode.
        
        Args:
            mode: 'train' or 'test'
            dataset: Dictionary containing dataset objects
            
        Returns:
            Dictionary containing data loaders
            
        Raises:
            ValueError: If dataset is None or mode is invalid
        """
        if dataset is None:
                raise ValueError('Dataset must be provided')
        if mode == 'train':
            train_dataset, val_dataset = dataset['train_dataset'], dataset['val_dataset']
            
            # Base loader configuration
            loader_config = {
                'batch_size': self.config['BATCH_SIZE'],
                'num_workers': self.config['NUM_WORKERS'],
                'pin_memory': True,
                'prefetch_factor': 4
            }
            
            # Create appropriate loaders based on training mode
            if self.config['TRAIN_MODE'] == TRAIN_MODE_BASE:
                train_loader = DataLoader(
                    train_dataset, 
                    shuffle=True, 
                    **loader_config
                )
                
            elif self.config['TRAIN_MODE'] == TRAIN_MODE_OVERSAMPLE:
                sampler = create_weighted_sampler(dataset['train_data']['rock_type'].values)
                train_loader = DataLoader(
                    train_dataset, 
                    sampler=sampler, 
                    **{k: v for k, v in loader_config.items() if k != 'shuffle'}
                )
                
            elif self.config['TRAIN_MODE'] == TRAIN_MODE_HARD_NEGATIVE:
                hard_negative_miner = HardNegativeMiner(
                    dataset_size=len(dataset['train_data']), 
                    memory_size=self.config['HARD_NEGATIVE_MEMORY_SIZE']
                )
                
                train_loader = create_train_loader_with_accumulation(train_dataset, hard_negative_miner, dataset['train_data']['rock_type'].values, dataset['num_classes'], 
                                        self.config['BATCH_SIZE'], self.config['ACCUMULATION_STEPS'], hard_negative_ratio=self.config['HARD_NEGATIVE_RATIO'], num_workers=self.config['NUM_WORKERS'])
                
                result = {
                    'train_loader': train_loader,
                    'val_loader': DataLoader(val_dataset, shuffle=False, **loader_config),
                    'hard_negative_miner': hard_negative_miner
                }
                return result
            
            elif self.config['TRAIN_MODE'] in [TRAIN_MODE_PROGRESSIVE_HARD_NEGATIVE]:
                hard_negative_miner = HardNegativeMiner(
                    dataset_size=len(dataset['train_data']), 
                    memory_size=self.config['HARD_NEGATIVE_MEMORY_SIZE']
                )
                train_loader = None # train_loader가 Training_Function 내에서 정의 됨
                result = {
                    'train_loader': train_loader,
                    'val_loader': DataLoader(val_dataset, shuffle=False, **loader_config),
                    'hard_negative_miner': hard_negative_miner
                }
                return result
            else:
                raise ValueError(f"Invalid training mode: {self.config['TRAIN_MODE']}")
            
            # Create validation loader (common for all modes except hard negative)
            val_loader = DataLoader(
                val_dataset, 
                shuffle=False, 
                **loader_config
            )
            
            return {'train_loader': train_loader, 'val_loader': val_loader}
            
        elif mode == 'test':
            test_dataset = dataset['test_dataset']
            test_loader = DataLoader(test_dataset, batch_size=self.config['BATCH_SIZE'], shuffle=False, num_workers=self.config['NUM_WORKERS'])
            return {'test_loader': test_loader}
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'train' or 'test'")
    
    def init_model(self, num_classes: int) -> torch.nn.Module:
        """
        Initialize the model based on configuration.
        
        Args:
            num_classes: Number of classes for the classification task
            
        Returns:
            Initialized PyTorch model
        """
        model_name = self.config['MODEL_NAME']
        model_type = MODEL_SOURCE_MAPPER.get(model_name)
        
        if model_type == 'huggingface':
            # Implementation for HuggingFace models
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            # Default to TIMM models
            model = timm.create_model(
                model_name, 
                pretrained=True, 
                num_classes=num_classes
            )
        
        return model
    
    def set_optimizer(self, model):
        optimizer_name = self.config.get('OPTIMIZER', 'adam')
        if optimizer_name == 'adam':
            return torch.optim.Adam(params=model.parameters(), lr=self.config.get("LEARNING_RATE", 3e-4))
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(params=model.parameters(), lr=self.config.get("LEARNING_RATE", 3e-4))
    
    def set_scheduler(self, optimizer):
        warmups = self.config.get('WARM_UP', 0)
        eta_min = self.config.get('ETA_MIN', 1e-8)
        scheduler = self.config.get('SCHEDULER', 'cosineannealing')
        if scheduler not in ['cosineannealing', 'cosine']:
            warnings.warn(
                f"Scheduler {scheduler} is not fully tested! Currently, only CosineAnnealingLR is fully supported. The CosineAnnealingLR will be used in this experiment",
                UserWarning)
        if warmups == 0:
            return CosineAnnealingLR(optimizer, T_max=self.config['EPOCHS'], eta_min=eta_min)
        else:
            warmup_scheduler = LinearLR(optimizer, start_factor=self.config.get("START_FACTOR", 1/3), end_factor=1.0, total_iters=warmups)

            # 2. Cosine Annealing
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=self.config['EPOCHS'] - warmups, eta_min=eta_min)

            # 3. Sequential 스케줄러
            return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmups])
        
    def set_experiment(self, cur_fold=None, idx=None) -> Dict[str, str]:
        """
        Configure experiment settings and save configuration.
        
        Returns:
            Dictionary containing experiment directory paths
        """
        trained_path = self.config.get('TRAINED_PATH', '')
        model_name = self.config['MODEL_NAME'].replace('.', '_')
        
        # Create experiment name and directory
        if not trained_path:
            if idx is None:
                idx = len([x for x in os.listdir(self.experiment_path) if x.startswith(model_name)])
            experiment_name = f"{model_name}_{idx + 1}" if self.config['FOLD'] == 0 else f"{model_name}_{idx + 1}_fold{cur_fold}"
        else:
            experiment_name = os.path.splitext(os.path.basename(trained_path))[0].split('-')[0]
        
        experiment_dir = os.path.join(self.experiment_path, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Create configuration dictionary
        config = {
            'experiment': {'name': experiment_name},
            'kFold': f"{cur_fold}/{self.config['FOLD']}",
            'model': {
                'name': self.config['MODEL_NAME'],
                'IMG_size': self.config['IMG_SIZE']
            },
            'train': {
                'epoch': self.config['EPOCHS'],
                'batch_size': self.config['BATCH_SIZE'],
                'lr': self.config['LEARNING_RATE'],
                'train_transform': [str(x) for x in self.config['TRAIN_TRANSFORM']],
                'optimizer': {
                    'name': self.optimizer.__class__.__name__
                },
                'scheduler': {
                    'name': self.scheduler.__class__.__name__
                },
                'hard_negative_ratio': self.config.get('HARD_NEGATIVE_RATIO', 0),
                'hard_negative_memory_size': self.config.get('HARD_NEGATIVE_MEMORY_SIZE', 0),
                'balanced_class_sampling': self.config['TRAIN_MODE'] in [TRAIN_MODE_OVERSAMPLE, TRAIN_MODE_HARD_NEGATIVE, TRAIN_MODE_PROGRESSIVE_HARD_NEGATIVE],
                'accumulation_steps': self.config.get('ACCUMULATION_STEPS', 1)
            },
            'validation': {
                'test_transform': [str(x) for x in self.config['TEST_TRANSFORM']]
            },
            'split': self.config['TEST_SIZE'],
            'seed': self.config['SEED'],
            'system': {
                'hostname': socket.gethostname(),
                'username': getpass.getuser(),
                'platform': platform.system(),
                'platform-release': platform.release(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
            }
        }
        if self.config['TRAIN_MODE'] in [TRAIN_MODE_HARD_NEGATIVE]:
            config['train']['effective_batch_size'] = self.config['BATCH_SIZE'] * self.config['ACCUMULATION_STEPS']
        
        # Add optimizer parameters
        for k, v in self.optimizer.state_dict()['param_groups'][0].items():
            if k == 'params':
                continue
            config['train']['optimizer'][k] = v
        
        # Add scheduler parameters
        for k, v in self.scheduler.state_dict().items():
            config['train']['scheduler'][k] = v
        
        # Save configuration
        config_path = os.path.join(experiment_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        return {'experiment_dir': experiment_dir, 'experiment_name': experiment_name, 'idx': idx}
    
    def _setup_logging(self, experiment_name: str) -> None:
        """
        Set up logging for the experiment.
        
        Args:
            experiment_name: Name of the experiment
        """
        # Initialize wandb logging
        wandb.init(
            project=self.config.get('WANDB_PROJECT', 'rock-classification'),
            name=experiment_name,
            entity = "alexseo-inha-university",
            # resume='must',
            # id='brs1qe2i',
            config={
                "model": self.config['MODEL_NAME'],
                "optimizer": self.config['OPTIMIZER'].__class__.__name__,
                "scheduler": self.config['SCHEDULER'].__class__.__name__,
                "epochs": self.config['EPOCHS'],
                "learning_rate": self.config['LEARNING_RATE'],
                "batch_size": self.config['BATCH_SIZE'],
                "img_size": self.config['IMG_SIZE'],
                "train_mode": self.config['TRAIN_MODE'],
                "hard_negative_ratio": self.config.get('HARD_NEGATIVE_RATIO', 0),
                "hard_negative_memory_size": self.config.get('HARD_NEGATIVE_MEMORY_SIZE', 0),
                "balanced_class_sampling": self.config['TRAIN_MODE'] in [TRAIN_MODE_OVERSAMPLE, TRAIN_MODE_HARD_NEGATIVE],
                "accumulation_steps": self.config.get('ACCUMULATION_STEPS', 1)
            }
        )
    
    def train(self) -> torch.nn.Module:
        """
        Run the training process.
        
        Returns:
            Trained model
        """
        if self.config['FOLD'] > 0:
            self.fold_train()
        else:
            self.single_train()
    
    def single_train(self) -> torch.nn.Module:
        """
        Run the training process.
        
        Returns:
            Trained model
        """
        # Initialize dataset
        dataset = self.init_dataset('train')
        
        # Initialize data loaders
        loaders = self.init_loader('train', dataset)
        train_loader, val_loader = loaders['train_loader'], loaders['val_loader']
        
        # Initialize model
        model = self.init_model(dataset['num_classes'])
        model = model.to(self.device)
        
        # Get training components
        optimizer = self.set_optimizer(model)
        scheduler = self.set_scheduler(optimizer)

        self.optimizer = optimizer # 로깅을 위함
        self.scheduler = scheduler # 로깅을 위함

        # Set up experiment
        exp_settings = self.set_experiment()
        experiment_name = exp_settings['experiment_name']
        experiment_dir = exp_settings['experiment_dir']

        # Set up logging
        self._setup_logging(experiment_name)
        
        start_epoch = 1
        best_score = 0.0
        if self.config.get('TRAINED_PATH', "") != "":
            checkpoint = torch.load(self.config.get('TRAINED_PATH'), map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_score = checkpoint['best_score']

        # Choose appropriate training function based on mode
        if self.config['TRAIN_MODE'] in [TRAIN_MODE_BASE, TRAIN_MODE_OVERSAMPLE]:
            # Standard training with optional oversampling
            print(f"Starting Standard Training ({self.config['TRAIN_MODE']})...")
            trained_model = train(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                scheduler=scheduler,
                device=self.device,
                cur_epoch=start_epoch,
                best_score=best_score,
                class_names=dataset['class_names'],
                experiment_name=experiment_name,
                folder_path=experiment_dir,
                accumulation_steps=self.config.get('ACCUMULATION_STEPS', 1),
                epochs=self.config['EPOCHS'],
                criterion=self._get_criterion()
            )
        elif self.config['TRAIN_MODE'] == TRAIN_MODE_HARD_NEGATIVE:
            # Hard negative sampling training
            print("Starting Training with Hard Negative Samples...")
            hard_negative_miner = loaders.get('hard_negative_miner')
            trained_model = hard_negative_train(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                scheduler=scheduler,
                device=self.device,
                cur_epoch=start_epoch,
                best_score=best_score,
                class_names=dataset['class_names'],
                hard_negative_miner=hard_negative_miner,
                experiment_name=experiment_name,
                folder_path=experiment_dir,
                accumulation_steps=self.config.get('ACCUMULATION_STEPS', 1),
                epochs=self.config['EPOCHS'],
                criterion=self._get_criterion()
            )
        elif self.config['TRAIN_MODE'] == TRAIN_MODE_PROGRESSIVE_HARD_NEGATIVE:
                progressive_scheduler = ProgressiveScheduler(
                                        initial_ratio=self.config['INITIAL_RATIO'],
                                        final_ratio=self.config['FINAL_RATIO'],
                                        total_epochs=self.config['EPOCHS'],
                                        schedule_type=self.config['SCHEDULE_TYPE']
                                    )
                hard_negative_miner = loaders.get('hard_negative_miner')
                trained_model= progressive_hard_negative_train(model, optimizer, dataset['train_dataset'], val_loader, scheduler, self.device, 
                                  dataset['class_names'], dataset['train_data']['rock_type'].values, dataset['num_classes'], self.config['BATCH_SIZE'], self.config.get('ACCUMULATION_STEPS', 1),
                                  progressive_scheduler, hard_negative_miner=hard_negative_miner, 
                                  criterion=self._get_criterion(), 
                                  best_score=0, epochs=self.config['EPOCHS'], cur_epoch=start_epoch, 
                                  experiment_name=experiment_name, folder_path=experiment_dir,
                                  num_workers=self.config['NUM_WORKERS'], prefetch_factor=4, pin_memory=True)
        else:
            raise ValueError(f"Invalid training mode: {self.config['TRAIN_MODE']}")
        
        # Close wandb logging
        wandb.finish()
        
        return trained_model
    
    def fold_train(self) -> List[torch.nn.Module]:
        """
        Run k-fold cross validation training process.
        
        Returns:
            List of trained models for each fold
        """
        # Initialize k-fold
        kFold_dataset = self.init_dataset('train') # dict 형태
        folder_idx = None
        
        print(f"Starting {self.config['FOLD']}-Fold Cross Validation Training...")

        start_fold = 0 if self.config['START_FOLD'] in [None, 0] else self.config['START_FOLD']
        end_fold = self.config['FOLD'] if self.config['END_FOLD'] in [None, self.config['FOLD']] else self.config['END_FOLD']

        for fold in range(start_fold, end_fold):
            dataset = kFold_dataset[fold]
            loaders = self.init_loader('train', dataset)
            train_loader, val_loader = loaders['train_loader'], loaders['val_loader']
            
            # Initialize model
            model = self.init_model(dataset['num_classes'])
            model = model.to(self.device)
            
            # Get training components
            optimizer = self.set_optimizer(model)
            scheduler = self.set_scheduler(optimizer)

            self.optimizer = optimizer # 로깅을 위함
            self.scheduler = scheduler # 로깅을 위함

            # Set up experiment
            exp_settings = self.set_experiment(fold, idx=folder_idx)
            experiment_name = exp_settings['experiment_name']
            experiment_dir = exp_settings['experiment_dir']
            folder_idx = exp_settings['idx']

            # Set up logging
            self._setup_logging(experiment_name)
            
            start_epoch = 1
            best_score = 0.0
            if self.config.get('TRAINED_PATH', "") != "":
                checkpoint = torch.load(self.config.get('TRAINED_PATH'), map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_score = checkpoint['best_score']

            # Choose appropriate training function based on mode
            if self.config['TRAIN_MODE'] in [TRAIN_MODE_BASE, TRAIN_MODE_OVERSAMPLE]:
                # Standard training with optional oversampling
                print(f"Starting Standard Training ({self.config['TRAIN_MODE']})...")
                trained_model = train(
                    model=model,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    scheduler=scheduler,
                    device=self.device,
                    cur_epoch=start_epoch,
                    best_score=best_score,
                    class_names=dataset['class_names'],
                    experiment_name=experiment_name,
                    folder_path=experiment_dir,
                    accumulation_steps=self.config.get('ACCUMULATION_STEPS', 1),
                    epochs=self.config['EPOCHS'],
                    criterion=self._get_criterion()
                )
            elif self.config['TRAIN_MODE'] == TRAIN_MODE_HARD_NEGATIVE:
                # Hard negative sampling training
                print("Starting Training with Hard Negative Samples...")
                hard_negative_miner = loaders.get('hard_negative_miner')
                trained_model = hard_negative_train(
                    model=model,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    scheduler=scheduler,
                    device=self.device,
                    cur_epoch=start_epoch,
                    best_score=best_score,
                    class_names=dataset['class_names'],
                    hard_negative_miner=hard_negative_miner,
                    experiment_name=experiment_name,
                    folder_path=experiment_dir,
                    accumulation_steps=self.config.get('ACCUMULATION_STEPS', 1),
                    epochs=self.config['EPOCHS'],
                    criterion=self._get_criterion()
                )

            elif self.config['TRAIN_MODE'] == TRAIN_MODE_PROGRESSIVE_HARD_NEGATIVE:
                progressive_scheduler = ProgressiveScheduler(
                                        initial_ratio=self.config['INITIAL_RATIO'],
                                        final_ratio=self.config['FINAL_RATIO'],
                                        total_epochs=self.config['EPOCHS'],
                                        schedule_type=self.config['SCHEDULE_TYPE']
                                    )
                hard_negative_miner = loaders.get('hard_negative_miner')
                trained_model= progressive_hard_negative_train(model, optimizer, dataset['train_dataset'], val_loader, scheduler, self.device, 
                                  dataset['class_names'], dataset['train_data']['rock_type'].values, dataset['num_classes'], self.config['BATCH_SIZE'], self.config.get('ACCUMULATION_STEPS', 1),
                                  progressive_scheduler, hard_negative_miner=hard_negative_miner, 
                                  criterion=self._get_criterion(), 
                                  best_score=0, epochs=self.config['EPOCHS'], cur_epoch=start_epoch, 
                                  experiment_name=experiment_name, folder_path=experiment_dir,
                                  num_workers=self.config['NUM_WORKERS'], prefetch_factor=4, pin_memory=True)
            else:
                raise ValueError(f"Invalid training mode: {self.config['TRAIN_MODE']}")
            
            # Close wandb logging
            wandb.finish()
            
        return trained_model

    def _get_criterion(self) -> torch.nn.Module:
        """
        Get the loss function based on configuration.
        
        Returns:
            Loss function
        """
        loss_type = self.config.get('LOSS_TYPE', 'CE')
        reduction = True if self.config['TRAIN_MODE'] in [TRAIN_MODE_BASE, TRAIN_MODE_OVERSAMPLE] else False
        label_smoothing = self.config.get('LABEL_SMOOTHING', 0)

        if loss_type == 'weighted_normalized':
            factor=self.config.get('FACTOR', 1)
            return weighted_normalized_CrossEntropyLoss(class_counts=self.dataset['class_counts'], label_smoothing=label_smoothing, factor=factor)
        elif loss_type == 'weighted_normalized_custom':
            return weighted_normalized_CrossEntropyLoss_custom(class_counts=self.dataset['class_counts'])
        elif loss_type == 'weighted_normalized_diff_weighted':
            return weighted_normalized_CrossEntropyLoss_diff_weighted(class_counts=self.dataset['class_counts'])
        elif reduction:
            return torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            return torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)
    
    def predict(self) -> pd.DataFrame:
        """
        Run prediction on test data.
        
        Args:
            model_path: Path to saved model checkpoint
            
        Returns:
            DataFrame with predictions
        """
        # Initialize test dataset and loader
        print("Preparing for Inference...")
        saved_name = self.config.get('TRAINED_PATH', None)
        if saved_name == None:
            raise SyntaxError('Trained Model Path is Missing')
        checkpoint = torch.load(saved_name, map_location=self.device)
        model = self.init_model(self.dataset['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])

        dataset = self.init_dataset('train')
        le = dataset['le']
        test_dataset = self.init_dataset('test')
        test_loader = self.init_loader('test', test_dataset)['test_loader']
        model.to(self.device).eval()
        preds = []
        print("Start Inferencing...")
        with torch.no_grad():
            for imgs in tqdm(iter(test_loader)):
                imgs = imgs.float().to(self.device)
                
                pred = model(imgs)
                
                preds += pred.argmax(1).detach().cpu().numpy().tolist()
        
        preds = le.inverse_transform(preds)

        submit = pd.read_csv('./sample_submission.csv')

        submit['rock_type'] = preds

        submit.to_csv(f"{os.path.splitext(saved_name)[0]}_submit_{checkpoint['epoch']}epoch.csv", index=False)
        return preds
import torch
import sklearn
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

import numpy as np
import wandb
import torch.nn.functional as F

from data_loaders.data_transformer import PatchDataTransformer
from torch.utils.data import DataLoader
from .task_handler import ClassificationHandler

class Trainer:
    '''
        @brief: Handle training for classification/regression. This class shouldn't know how to generate the loader
                We need to pass the loaders. Caller should be responsible in supplying the data to be used
        @params:
            - model_confgs: all the models we want to train with its config
    '''
    def __init__(
        self,
        task: str,
        model_configs: [dict],
        train_loader: DataLoader,
        test_loader: DataLoader,
        val_loader=None
    ):
        self.task = task
        self.model_configs = model_configs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.is_val_available = val_loader is not None
        
        self.handler = ClassificationHandler()
            
    def _get_transformer(self, loader):
        dataset = loader.dataset
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        
        if hasattr(dataset, 'transformer'):
            return dataset.transformer
        return None

    def _update_loader_config(self, loader, config):
        dataset = loader.dataset
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        
        dataset.update_transformer(config)

    def execute(self):
        '''
            @brief: We currently have 2 classification training
                    - Linear Probing: Encoder extract -> ML models (no training loop explicitly)
                    - Light Fine-tune: Encoder + Head (epochs based training loop)
        ''' 

        '''
            configs:
            [
                {
                    classifier: c,
                    encoder: e,
                    use_encoder: bool,
                    patchify: bool
                },
                {
                    classifier: c,
                    encoder: e,
                    use_encoder: bool
                },
            ]
        '''
        all_results = []
        for config in self.model_configs:
            # Update loaders with current config
            self._update_loader_config(self.train_loader, config)
            self._update_loader_config(self.test_loader, config)
            if self.is_val_available:
                self._update_loader_config(self.val_loader, config)

            # parse config
            name = config.get("name", None)
            assert name, "ERROR: No name found in config"

            classifier = config.get("classifier", None)
            assert classifier is not None, "ERROR: No classifier found in config"

            train_type = config.get("train_type", None)
            assert train_type, "ERROR: Define train type"

            use_encoder = config.get("use_encoder", False)
            patchify = config.get('patchify', False)
            encoder = None
            if use_encoder:
                encoder = config.get("encoder", None)
                assert encoder, "ERROR: Expected to find encoder with use_encoder = True"

            if train_type == "classical":
                # (x_list numpy, y_list)
                train_tuple, test_tuple, val_tuple = self.aggregate_data(patchify=patchify)

                # encode all the raw input
                if use_encoder:
                    # freeze encoder
                    for param in encoder.parameters():
                        param.requires_grad = False
                    encoder.eval()
                    train_tuple = self.extract(encoder, train_tuple)
                    test_tuple = self.extract(encoder, test_tuple)
                    if self.is_val_available:
                        val_tuple = self.extract(encoder, val_tuple)
                else:
                    # apply PCA
                    pca = sklearn.decomposition.PCA(n_components=2)
                    train_tuple = (pca.fit_transform(train_tuple[0]), train_tuple[2])
                    test_tuple = (pca.transform(test_tuple[0]), test_tuple[2])

                    if self.is_val_available:
                        val_tuple = (pca.transform(val_tuple[0]), val_tuple[2])

                results = self.train_classical_ml(
                    classifier,
                    train_tuple,
                    test_tuple,
                    val_tuple,
                    name=name
                )
                results.update({"name": name})
                all_results.append(results)
            elif train_type == "dl":
                results = self.train_dl(
                    model=classifier,
                    encoder=encoder,
                    lr=config["lr"],
                    batch_size=config["batch_size"],
                    num_epochs=config["num_epochs"],
                    fine_tune_type=config["fine_tune_type"]
                )
                results.update({"name": name})
                all_results.append(results)

        return all_results

    def extract(self, encoder, data_tuple):
        '''
        @brief: Extract features from encoder and aggregate patches
        @params:
            - encoder: The encoder model
            - data_tuple: Tuple of (x, y) where x is input data
        @return: Tuple of (encoded_features, y) where encoded_features are numpy arrays
        '''
        x, x_mark, y = data_tuple
        transformer = self._get_transformer(self.train_loader)
        
        # Ensure x is on the same device as encoder
        device = next(encoder.parameters()).device
        x = x.to(device)
        x_mark = x_mark.to(device)
        
        if isinstance(transformer, PatchDataTransformer):
            encoded = transformer.encode(encoder, x, x_mark)
        else:
            encoded = transformer.encode(encoder, x)
            
        encoded = encoded.detach().cpu().numpy()
        return (encoded, y)

    def train_dl(self, 
        model, 
        encoder,
        lr,
        batch_size,
        num_epochs,
        fine_tune_type,
    ):
        # Ensure model is on the same device as encoder
        device = next(encoder.parameters()).device
        model = model.to(device)

        for param in encoder.parameters():
            if fine_tune_type == "full":
                param.requires_grad = True
            else:
                # freeze encoder
                param.requires_grad = False
        
        param_groups = [{"params": (p for p in model.parameters()), "lr": lr}]
        if fine_tune_type == "full":
            param_groups.append({"params": (p for p in encoder.parameters()), "lr": lr})

        optimizer = torch.optim.AdamW(param_groups)
        criterion = self.handler.get_criterion()

        train_logs = {}

        for e in range(num_epochs):
            encoder.eval()
            model.train()

            train_loss = 0
            train_logits = []
            train_labels = []

            test_loss = 0
            test_logits = []
            test_labels = []

            # Get transformer for the current loader
            transformer = self._get_transformer(self.train_loader)

            for subjects_trains, x_trains, y_trains in self.train_loader:
                optimizer.zero_grad()
                
                # Ensure x is on the same device as encoder
                x_trains = x_trains.to(device)
                y_trains_device = y_trains.to(device)
                
                mean_embedding = transformer.encode(encoder, x_trains)
                predicted_logits = model(mean_embedding)
                loss = criterion(
                    predicted_logits.squeeze(), y_trains_device.float()
                )

                # collect for metrics
                train_logits.append(predicted_logits.detach().cpu())
                train_labels.append(y_trains)

                loss.backward()
                optimizer.step()

                train_loss += loss / batch_size
            
            train_preds, train_probs = self.handler.process_logits(train_logits)
            train_labels = torch.cat(train_labels)
            train_metrics = self.handler.get_metrics(train_preds, train_labels, train_probs)

            # ===========
            # Evaluation
            # ===========
            with torch.no_grad():
                # Get transformer for test loader
                test_transformer = self._get_transformer(self.test_loader)
                
                for subjects_test, x_tests, y_tests in self.test_loader:
                    encoder.eval()
                    model.eval()  

                    x_tests = x_tests.to(device)
                    y_tests_device = y_tests.to(device)

                    mean_embedding = test_transformer.encode(encoder, x_tests)
                    predicted_logits = model(mean_embedding)
                    loss = criterion(predicted_logits.squeeze(), y_tests_device.float())

                    test_loss += loss / batch_size
                    test_logits.append(predicted_logits.cpu())
                    test_labels.append(y_tests)
                
                test_preds, test_probs = self.handler.process_logits(test_logits)
                test_labels = torch.cat(test_labels)
                test_metrics = self.handler.get_metrics(test_preds, test_labels, test_probs)

                val_metrics = None
                if self.is_val_available:
                    val_loss = 0
                    val_logits = []
                    val_labels = []
                    
                    val_transformer = self._get_transformer(self.val_loader)
                    
                    for subjects_val, x_vals, y_vals in self.val_loader:
                        x_vals = x_vals.to(device)
                        y_vals_device = y_vals.to(device)
                        
                        mean_embedding = val_transformer.encode(encoder, x_vals)
                        predicted_logits = model(mean_embedding)
                        loss = criterion(predicted_logits.squeeze(), y_vals_device.float())

                        val_loss += loss / batch_size
                        val_logits.append(predicted_logits.cpu())
                        val_labels.append(y_vals)
                    
                    val_preds, val_probs = self.handler.process_logits(val_logits)
                    val_labels = torch.cat(val_labels)
                    val_metrics = self.handler.get_metrics(val_preds, val_labels, val_probs)

                if e % 1 == 0 and train_metrics is not None:
                    print(f"Epoch: {e} - Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
                    print(f"Train Metrics: {train_metrics}")
                    print(f"Test Metrics: {test_metrics}")
                    if val_metrics:
                        print(f"Val Metrics: {val_metrics}")
                
                train_logs[e] = {
                    "train": train_metrics,
                    "test": test_metrics,
                    "val": val_metrics if self.is_val_available else {}
                }

        return {
            "train": train_metrics,
            "test": test_metrics,
            "val": val_metrics if self.is_val_available else {}
        }

    def train_classical_ml(
        self, 
        model,
        train: (list, list), 
        test: (list, list), 
        val: (list, list),
        name: str = "model"
    ):
        y_train = np.array(train[1])
        y_test = np.array(test[1])
        y_val = np.array(val[1])
        
        model.fit(train[0], y_train)
        y_train_preds = self.handler.predict(model, train[0])
        y_train_probs = self.handler.predict_proba(model, train[0])

        y_test_preds = self.handler.predict(model, test[0])
        y_test_probs = self.handler.predict_proba(model, test[0])

        if self.is_val_available:
            y_val_preds = self.handler.predict(model, val[0])
            y_val_probs = self.handler.predict_proba(model, val[0])
        
        train_metrics = self.handler.get_metrics(y_train_preds, y_train, y_train_probs)
        test_metrics = self.handler.get_metrics(y_test_preds, y_test, y_test_probs)
        val_metrics = self.handler.get_metrics(y_val_preds, y_val, y_val_probs) if self.is_val_available else {}
        
        results = {
            "train": train_metrics,
            "test": test_metrics,
            "val": val_metrics
        }

        return results

    def aggregate_data(self, patchify):
        x_train_all = []
        x_train_mark_all = []
        y_train_all = []

        x_test_all = []
        x_test_mark_all = []
        y_test_all = []

        for subjects_trains, x_train_patches_tensors, x_train_mark_patches_tensors, y_trains in self.train_loader:
            x_trains = x_train_patches_tensors
            x_trains_mark = x_train_mark_patches_tensors
            if not patchify:
                x_trains = x_train_patches_tensors.view(x_train_patches_tensors.size(0), -1)
                x_trains_mark = x_train_mark_patches_tensors.view(x_train_mark_patches_tensors.size(0), -1)
            x_train_all.extend(x_trains.cpu().detach().numpy())
            x_train_mark_all.extend(x_trains_mark.cpu().detach().numpy())
            y_train_all.extend(y_trains)
        x_train_all = torch.Tensor(x_train_all)
        x_train_mark_all = torch.Tensor(x_train_mark_all)

        for subjects_tests, x_test_patches_tensors, x_test_mark_patches_tensors, y_tests in self.test_loader:
            x_tests = x_test_patches_tensors
            x_tests_mark = x_test_mark_patches_tensors
            if not patchify:
                x_tests = x_test_patches_tensors.view(x_test_patches_tensors.size(0), -1)
                x_tests_mark = x_test_mark_patches_tensors.view(x_test_mark_patches_tensors.size(0), -1)
            x_test_all.extend(x_tests.cpu().detach().numpy())
            x_test_mark_all.extend(x_tests_mark.cpu().detach().numpy())
            y_test_all.extend(y_tests)
        x_test_all = torch.Tensor(x_test_all)
        x_test_mark_all = torch.Tensor(x_test_mark_all)

        x_val_all = None
        x_val_mark_all = None
        y_val_all = None
        if self.is_val_available:
            x_val_all = []
            x_val_mark_all = []
            y_val_all = []
            for subjects_vals, x_val_patches_tensors, x_val_mark_patches_tensors, y_vals in self.val_loader:
                x_vals = x_val_patches_tensors
                x_vals_mark = x_val_mark_patches_tensors
                if not patchify:
                    x_vals = x_val_patches_tensors.view(x_val_patches_tensors.size(0), -1)
                    x_vals_mark = x_val_mark_patches_tensors.view(x_val_mark_patches_tensors.size(0), -1)
                x_val_all.extend(x_vals.cpu().detach().numpy())
                x_val_mark_all.extend(x_vals_mark.cpu().detach().numpy())
                y_val_all.extend(y_vals)
            x_val_all = torch.Tensor(x_val_all)
            x_val_mark_all = torch.Tensor(x_val_mark_all)

        return (x_train_all, x_train_mark_all, y_train_all), (x_test_all, x_test_mark_all, y_test_all), (x_val_all, x_val_mark_all, y_val_all)


import numpy as np
from tqdm import tqdm
import os

from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader
from typing import Any, Optional, List, Union, Tuple
import logging
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, top_k_accuracy_score
from itertools import chain
from logadempirical.data.log import Log

from logadempirical.data.dataset import LogDataset


class Trainer:
    def __init__(self, model,
                 train_dataset: LogDataset,
                 valid_dataset: LogDataset,
                 is_train=True,
                 optimizer: torch.optim.Optimizer = None,
                 no_epochs: int = 100,
                 batch_size: int = 32,
                 scheduler_type: str = 'linear',
                 warmup_rate: float = 0.1,
                 accumulation_step: int = 1,
                 decay_rate: float = 0.9,
                 logger: logging.Logger = None,
                 accelerator: Any = None,
                 num_classes: int = 2,
                 ):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.is_train = is_train
        self.optimizer = optimizer
        self.no_epochs = no_epochs
        self.batch_size = batch_size
        self.scheduler_type = scheduler_type
        self.warmup_rate = warmup_rate
        self.accumulation_step = accumulation_step
        self.decay_rate = decay_rate
        self.logger = logger
        self.accelerator = accelerator
        self.num_classes = num_classes
        self.scheduler = None

    def _train_epoch(self,
                     train_loader: DataLoader,
                     device: str,
                     scheduler: Any,
                     progress_bar: Any):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0
        for idx, batch in enumerate(train_loader):
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = self.model(batch, device=device)
            loss = outputs.loss
            total_loss += loss.item()
            loss = loss / self.accumulation_step
            self.accelerator.backward(loss)
            if (idx + 1) % self.accumulation_step == 0 or idx == len(train_loader) - 1:
                self.optimizer.step()
                self.optimizer.zero_grad()
                scheduler.step()
                progress_bar.update(1)
                progress_bar.set_postfix({'loss': total_loss / (idx + 1)})

        return total_loss / len(train_loader)

    def _valid_epoch(self,
                     val_loader: DataLoader,
                     device: str,
                     topk: int = 1):
        self.model.eval()
        y_pred = []
        y_true = []
        losses = []
        for idx, batch in enumerate(val_loader):
            del batch['idx']
            # batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(batch, device=device)
            loss = outputs.loss
            probabilities = self.accelerator.gather(outputs.probabilities)
            y_pred.append(probabilities.detach().clone().cpu().numpy())
            # y_pred = self.accelerator.gather(y_pred)
            losses.append(loss.item())
            label = self.accelerator.gather(batch['label'])
            y_true.append(label.detach().clone().cpu().numpy())
        # concatenate because there are arrays of arrays
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)

        loss = np.mean(losses)

        if topk > 1:
            for k in range(1, self.num_classes + 1):
                acc = top_k_accuracy_score(
                    y_true, y_pred, k=k, labels=np.arange(self.num_classes))
                if acc >= 0.997:
                    self.logger.info(f"Top-{k} accuracy: {acc}")
                    return loss, acc, k
        else:
            acc = accuracy_score(y_true, np.argmax(y_pred, axis=1))
        return loss, acc, 1

    def train(self,
              device: str = 'cpu',
              save_dir: str = None,
              model_name: str = None,
              topk: int = 9):
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(
            self.valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.model.to(device)
        self.model, self.optimizer, train_loader, val_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader
        )
        num_training_steps = int(
            self.no_epochs * len(train_loader) / self.accumulation_step)
        num_warmup_steps = int(num_training_steps * self.warmup_rate)
        self.scheduler = get_scheduler(
            self.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        progress_bar = tqdm(range(num_training_steps), desc=f"Training",
                            disable=not self.accelerator.is_local_main_process)
        total_train_loss = 0
        total_val_loss = 0
        total_val_acc = 0

        for epoch in range(self.no_epochs):
            train_loss = self._train_epoch(
                train_loader, device, self.scheduler, progress_bar)
            val_loss, val_acc, valid_k = self._valid_epoch(
                val_loader, device, topk=topk)
            if self.logger is not None:
                self.logger.debug(
                    f"Epoch {epoch + 1}||Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            total_train_loss += train_loss
            total_val_loss += val_loss
            total_val_acc += val_acc
            if save_dir is not None and model_name is not None:
                self.save_model(save_dir, model_name)
        _, _, train_k = self._valid_epoch(train_loader, device, topk=topk)
        self.logger.info(
            f"Train top-{topk}: {train_k}, Valid top-{topk}: {valid_k}")
        self.save_model(save_dir, model_name)
        return total_train_loss / self.no_epochs, val_loss, val_acc, max(train_k, valid_k)
        # return total_train_loss / self.no_epochs, val_loss, val_acc, topk

    def predict_unsupervised(self,
                             dataset: LogDataset,
                             y_true,
                             topk: int,
                             device: str = 'cpu',
                             is_valid: bool = False,
                             num_sessions: Optional[List[int]] = None,
                             eventIds: Optional[List[str]] = None,
                             storeLog: Log = None
                             ) -> Union[Tuple[float, float, float, float], Tuple[float, int]]:
        def find_topk(dataloader):
            y_topk = []
            torch.set_printoptions(threshold=torch.inf)
            for batch in dataloader:
                # batch = {k: v.to(device) for k, v in batch.items()}
                label = self.accelerator.gather(batch['label'])

                with torch.no_grad():
                    # here not using topk because we want to find the position of the label in the sorted array
                    y_prob = self.model.predict(batch, device=device)

                y_pred = torch.argsort(y_prob, dim=1, descending=True)
                y_pos = torch.where(y_pred == label.unsqueeze(1))[1] + 1

                y_topk.extend(y_pos.cpu().numpy().tolist())
            return int(np.ceil(np.percentile(y_topk, 0.99)))

        test_loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False)
        self.model.to(device)
        self.model, test_loader = self.accelerator.prepare(
            self.model, test_loader)
        self.model.eval()
        if is_valid:
            acc, _, _, _ = self.predict_unsupervised_helper(
                test_loader, y_true, topk, device)
            return acc, find_topk(test_loader)
        else:
            return self.predict_unsupervised_helper(test_loader, y_true, topk, device, num_sessions, eventIds, storeLog=storeLog)

    def predict_unsupervised_helper(self, test_loader, y_true, topk: int, device: str = 'cpu',
                                    num_sessions: Optional[List[int]] = None,
                                    eventIds: Optional[List[str]] = None,
                                    storeLog=Log,
                                    ) -> Tuple[float, float, float, float]:
        y_pred = {k: 0 for k in y_true.keys()}
        # unknown_sequences = []
        count_unk_events = {k: 0 for k in y_true.keys()}
        count_predicted_anomalies = {k: 0 for k in y_true.keys()}
        progress_bar = tqdm(total=len(test_loader), desc=f"Predict",
                            disable=not self.accelerator.is_local_main_process)

        print(f"Total labeled: {len(y_true)}")
        for batch in test_loader:
            idxs = self.accelerator.gather(
                batch['idx']).detach().clone().cpu().numpy().tolist()
            support_label = (batch['sequential'] >=
                             self.num_classes).any(dim=1)

            sequentials = self.accelerator.gather(
                batch['sequential']).detach().clone().cpu().numpy().tolist()

            support_label = self.accelerator.gather(
                support_label).cpu().numpy().tolist()
            batch_label = self.accelerator.gather(
                batch['label']).cpu().numpy().tolist()

            del batch['idx']

            with torch.no_grad():
                y = self.accelerator.unwrap_model(self.model).predict_class(
                    batch, top_k=topk, device=device)
            y = self.accelerator.gather(y).cpu().numpy().tolist()

            # idxs is a list of indices representing sessions or sequences.
            # y is a list of lists containing the top-k predicted labels for each batch.
            # batch_label is a list that contains the next event label for each batch.
            # support_label is a list of Boolean values indicating whether each batch contains an unknown event.

            for idx, y_i, label_i, s_label in zip(idxs, y, batch_label, support_label):
                # if the ground truth label is not among the top-k predicted labels for that session or if the session contains an unknown event then the session is labeled as an anomaly
                if s_label == 1 and y_pred[idx] == 0:
                    y_pred[idx] = s_label
                    count_unk_events[idx] = s_label
                    count_predicted_anomalies[idx] = 0
                elif y_pred[idx] == 0:
                    y_pred[idx] = y_pred[idx] | (label_i not in y_i)
                    count_predicted_anomalies[idx] = y_pred[idx]
                    count_unk_events[idx] = 0

                # sequentials_idx = [seq for k, seq in enumerate(
                #     sequentials) if idxs[k] == idx]

                # if prediction is normal and the session contains an unknown event then get all unknown events which are labeled as anomalies
                # if s_label and eventIds[idx] not in unknown_sequences and label_i in y_i:
                #     # TODO if new event in w=10, len(arr)=15, steps = 5 then it may be repeated {1-5} times, this can be optimized
                #     for k, seq in enumerate(sequentials_idx):
                #         if self.num_classes in seq and eventIds[idx] not in unknown_sequences:
                #             # k is the sequence step, seq is the sequence and eventIds[idx] is the SessionId. We want to find the new events that are not in the vocabulary and when they appear.
                #             unknown_sequences.append(eventIds[idx])

            progress_bar.update(1)
        progress_bar.close()
        idxs = list(y_pred.keys())
        self.logger.info(f"Computing metrics...")

        if num_sessions is not None:
            self.logger.info(f"Total sessions: {sum(num_sessions)}")
            # --TESTING ONLY------
            print("Number of sessions: ", num_sessions)
            print(
                f"\ncount unk events: {count_unk_events} total {sum(count_unk_events.values())}")
            print(
                f"count predicted: {count_predicted_anomalies} total {sum(count_predicted_anomalies.values())}")
            test = np.array([v for _, v in y_true.items()])
            print("real anomalies: ", test, " total ", sum(test), "\n")
            test2 = np.array([v for _, v in count_predicted_anomalies.items()])
            test3 = np.array([v for _, v in count_unk_events.items()])
            missed_anomalies_indices = np.where(
                (test == 1) & (test2 == 0))[0]
            print("missed anomalies indices: ", missed_anomalies_indices)
            unk_anomalies_indices = np.where(
                (test == 1) & (test3 == 1))[0]
            print("unk indices: ", unk_anomalies_indices, "\n")
            # -----------------
            eventIds = [eventIds[idx] for idx in idxs]
            eventIds_replicated = np.concatenate(
                [np.repeat(e, n) for e, n in zip(eventIds, num_sessions)])
            y_pred = [[y_pred[idx]] * num_sessions[idx] for idx in idxs]
            y_true = [[y_true[idx]] * num_sessions[idx] for idx in idxs]
            count_predicted_anomalies = [
                [count_predicted_anomalies[idx]] * num_sessions[idx] for idx in idxs]
            count_unk_events = [[count_unk_events[idx]]
                                * num_sessions[idx] for idx in idxs]

            y_pred = np.array(list(chain.from_iterable(y_pred)))
            y_true = np.array(list(chain.from_iterable(y_true)))
            count_predicted_anomalies = np.array(
                list(chain.from_iterable(count_predicted_anomalies)))
            count_unk_events = np.array(
                list(chain.from_iterable(count_unk_events)))

            # Find indices where y_true is equal to 1 after num_sessions replication
            total_real_anomalies = np.where(y_true == 1)[0]
            # Find indices where both y_true and y_pred are equal to 1
            true_positives = np.where((y_true == 1)
                                      & (y_pred == 1))[0]

            y_pred_anomalies = np.where(y_pred == 1)[0]

            print(f"unk events * sess : {count_unk_events}")
            print(f"count predicted * sess: {count_predicted_anomalies}")

            count_unk_events = np.where(np.array(count_unk_events) == 1)[0]
            count_predicted_anomalies = np.where(
                np.array(count_predicted_anomalies) == 1)[0]

            print(f"\nunknown events: {count_unk_events}")
            print(f"predicted anomalies: {count_predicted_anomalies}")

            print(
                f"[Total_real_anomalies/true positives: {len(total_real_anomalies)} / {len(true_positives)}] [unknown events: {len(count_unk_events)}] [predicted anomalies: {len(count_predicted_anomalies)}] [total_pred: {len(y_pred_anomalies)}]\n")
            # See anomalies as original logs
            print("Found anomalies: \n")
            for idx in y_pred_anomalies:

                original_log = storeLog.get_original_data(
                    eventIds_replicated[idx])
                if idx in count_unk_events:
                    print(f"unk: {idx} at: { eventIds_replicated[idx]}")
                elif idx in count_predicted_anomalies:
                    print(f"pred: {idx} at: { eventIds_replicated[idx]}")
                else:
                    print(
                        f"not detected: {idx} at: { eventIds_replicated[idx]}")
        else:
            y_pred = np.array([y_pred[idx] for idx in idxs])
            y_true = np.array([y_true[idx] for idx in idxs])

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        progress_bar.close()
        return acc, f1, pre, rec

    def train_on_false_positive(self,
                                false_positive_dataset: LogDataset,
                                device: str = 'cpu',
                                save_dir: str = None,
                                model_name: str = None,
                                topk: int = 9):
        train_loader = DataLoader(
            false_positive_dataset, batch_size=self.batch_size, shuffle=False)
        self.model.to(device)
        self.model, train_loader = self.accelerator.prepare(
            self.model, train_loader)
        self.optimizer.zero_grad()
        # Train the model on the false positive anomaly data
        num_training_steps = int(
            self.no_epochs * len(train_loader) / self.accumulation_step)
        num_warmup_steps = int(num_training_steps * self.warmup_rate)
        self.scheduler = get_scheduler(
            self.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        progress_bar = tqdm(range(num_training_steps), desc=f"Training",
                            disable=not self.accelerator.is_local_main_process)
        total_train_loss = 0

        for epoch in range(self.no_epochs):
            print(f"Epoch {epoch + 1}")
            train_loss = self._train_epoch(
                train_loader, device, self.scheduler, progress_bar)
            total_train_loss += train_loss
        if save_dir is not None and model_name is not None:
            print(f"Saving model to {save_dir}/{model_name}.pt")
            self.save_model(save_dir, model_name)
        _, _, train_k = self._valid_epoch(train_loader, device, topk=topk)
        print(
            f"total_train_loss: {total_train_loss / self.no_epochs} top-{topk}: {train_k}")
        return total_train_loss / self.no_epochs, train_k

    def save_model(self, save_dir: str, model_name: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save(
            {
                "lr_scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "optimizer": self.optimizer.state_dict(),
                "model": self.model.state_dict()
            },
            f"{save_dir}/{model_name}.pt"
        )

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path)

        self.model = self.accelerator.unwrap_model(self.model)
        # self.model.load_state_dict(checkpoint['model'])
        self.model.load_state_dict(checkpoint['model'])

        self.model = self.accelerator.prepare(self.model)
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

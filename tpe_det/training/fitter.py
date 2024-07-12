import torch
import os
import time
from datetime import datetime
from glob import glob
from evaluation.metrics import count_metrics
from utils.markings import get_all_marking, get_predicted_marking_validation
from utils.average_meter import AverageMeter

class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5

        self.model = model
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(
            self.optimizer, **config.scheduler_params)
        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        summary_loss_over_itr_train = []
        summary_loss_over_itr_val = []
        history = []
        all_marking = get_all_marking(validation_loader.dataset)
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss, step_history = self.train_one_epoch(train_loader)
            history.append(step_history)
            summary_loss_over_itr_train.append(summary_loss)
            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss, precision_ave, recall_ave = self.validation(
                validation_loader, all_marking)
            summary_loss_over_itr_val.append(summary_loss)

            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}, precision: {precision_ave:.5f}, recall: {recall_ave:.5f}')
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(
                    f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1
        return self.best_summary_loss, summary_loss_over_itr_train, summary_loss_over_itr_val, history

    def validation(self, val_loader, all_marking):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        precision_list = []
        recall_list = []
        for step, (images, targets, image_ids, _) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' +
                        f'summary_loss: {summary_loss.avg:.5f}, ' +
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                batch_size = len(images)
                images = torch.stack(images).to(self.device).float()
                boxes = [target['boxes'].to(self.device).float()
                         for target in targets]
                labels = [target['labels'].to(
                    self.device).float() for target in targets]

                for i in range(len(images)):
                    img = images[i].unsqueeze(0)
                    bbox = boxes[i]
                    cls = labels[i]

                    if bbox.nelement() == 0 or cls.nelement() == 0:
                        continue

                    target = {
                        "bbox": bbox.unsqueeze(0),
                        "cls": cls.unsqueeze(0),
                        'img_scale': None,
                        'img_size': None,
                    }

                    # loss, _, _, _ = self.model(img, target)
                    score_threshold = 0.05
                    output = self.model(img, target)
                    predicted_marking = get_predicted_marking_validation(
                        output['detections'][0], i, image_ids[0], score_threshold=score_threshold)
                    if len(predicted_marking) == 0:
                        continue
                    result = count_metrics(all_marking, predicted_marking)

                    try:
                        precision = result[3]/(result[3] + result[1])
                    except ZeroDivisionError:
                        precision = 0.0

                    try:
                        recall = result[3]/(result[3] + result[5])
                    except ZeroDivisionError:
                        recall = 0.0

                    precision_list.append(precision)
                    recall_list.append(recall)
                    summary_loss.update(
                        output["loss"].detach().item(), batch_size)

        try:
            precision_ave = sum(precision_list) / len(precision_list)
        except ZeroDivisionError:
            precision_ave = 0.0
        try:
            recall_ave = sum(recall_list) / len(recall_list)
        except ZeroDivisionError:
            recall_ave = 0.0
        return summary_loss, precision_ave, recall_ave

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        history = []

        for step, (images, targets, _, _) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' +
                        f'summary_loss: {summary_loss.avg:.5f}, ' +
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            batch_size = len(images[0])
            images = [image.to(self.device).float() for image in images]
            boxes = [target['boxes'].to(self.device).float()
                     for target in targets]
            labels = [target['labels'].to(self.device).float()
                      for target in targets]
            self.optimizer.zero_grad()

            for i in range(len(images)):
                img = images[i].unsqueeze(0)
                bbox = boxes[i]
                cls = labels[i]

                # Check if the current slice has any bounding boxes
                if bbox.nelement() == 0 or cls.nelement() == 0:
                    continue

                target = {
                    "bbox": boxes[i].unsqueeze(0),
                    "cls": labels[i].unsqueeze(0)
                }

                output = self.model(img, target)
                output['loss'].backward()
                summary_loss.update(output['loss'].detach().item(), batch_size)

                # TODO: change with appropriate metrics
                history.append(summary_loss.avg)

            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()

        return summary_loss, history

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
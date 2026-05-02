import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer:
    def __init__(self, model, train_dataset, eval_dataset, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
        )
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
        )
        self.optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config["epochs"])

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            _, loss = self.model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        self.scheduler.step()
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in self.eval_loader:
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, targets=y)
                total_loss += loss.item()
        return total_loss / len(self.eval_loader)

    def save_checkpoint(self, path: str, epoch: int):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )

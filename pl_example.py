## test example
class MnistModel(pl.LightningModule):
    def __init__(self, hparams=dict(train_batch_size=10, lr=.1, milestones=[2, 4, 6], gamma=.5,
                      hidden_units=20, num_workers=4, val_batch_size=100,
                      train_size=10000)):

        super().__init__()
        self.hparams = hparams
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=torch.tensor(0.1307),
                                             std=torch.tensor(0.3081))
        ])

        ds_train = torchvision.datasets.MNIST(root='/nvme_drive/orm/MNIST',
                                              download=True, train=True,
                                              transform=self.transforms)

        indices = np.random.permutation(len(ds_train))[:self.hparams['train_size']]
        self.ds_train = torch.utils.data.Subset(ds_train, indices)
        self.ds_val = torchvision.datasets.MNIST(root='/nvme_drive/orm/MNIST/', download=True, train=False,
                                                 transform=self.transforms)

        num_hidden = hparams['hidden_units']
        self.model = nn.Sequential(
            nn.Conv2d(1, num_hidden, (3, 3), 1, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True),
            nn.Conv2d(num_hidden, num_hidden, (3, 3), 1, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_hidden, 10)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def loss(self, out, target):
        return self.loss_fn(out, target)

    def forward(self, X):
        return self.model(X)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_train,
                                           batch_size=self.hparams['train_batch_size'], shuffle=True,
                                           num_workers=self.hparams['num_workers'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_val,
                                           batch_size=self.hparams['val_batch_size'], shuffle=False,
                                           num_workers=self.hparams['num_workers'])

    def configure_optimizers(self):
        opt = torch.optim.SGD(params=self.parameters(), lr=self.hparams['lr'])
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=self.hparams['milestones'],
                                                     gamma=self.hparams['gamma'], last_epoch=-1)
        return [opt], [sched]

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        data, target = batch
        out = self.model(data)
        loss = self.loss(out, target)
        return {'loss': loss, 'log': {'loss': loss}}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        out = self.model(data)
        losses = self.loss(out, target)
        accuracy = (out.max(dim=-1).indices == target)
        return {'val_loss': losses, 'val_acc': accuracy}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([o['val_loss'] for o in outputs]).mean()
        acc = torch.cat([o['val_acc'] for o in outputs]).float().mean()
        return {'val_loss': loss, 'log': {'val_acc': acc, 'val_loss': loss}}

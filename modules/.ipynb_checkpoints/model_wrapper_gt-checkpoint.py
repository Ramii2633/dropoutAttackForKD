import time
from model_wrapper import NetWrapper


class NetWrapper_T(NetWrapper):
    def __init__(self, model, loss_fcn, optimizer, optim_args, reshape_fcn=None):
        super(NetWrapper_T, self).__init__(model, loss_fcn, optimizer, optim_args, reshape_fcn)
    
    def fit(self, train_input, val_input, target_class, num_epochs, verbose, attack_epoch=None, num_classes=10):
        """
        Train the model

            Parameters:
                train_input: a dataloader to train the model on
                val_input: a dataloader to run model validation
                target_class: number representing the target class to use
                num_epochs: number of epochs to run
                verbose: whether or not model statistics are printed
                attack_epoch: when to start the attack
        """
        for i in range(num_epochs):
            start = time.time()
            train_loss = 0
            self.model.train()
            for _, (data, target) in enumerate(train_input):
                if self.reshape_fcn is not None:
                    data = self.reshape_fcn(data)                   

                data = data.to(device=self.device)
                target = target.to(device=self.device)
                self.optimizer.zero_grad()
                if attack_epoch is None or (attack_epoch is not None and i >= attack_epoch):
                    output = self.model.forward(data, target, target_class, True)
                else:
                    output = self.model.forward(data, target, target_class, False)
                loss = self.loss_fcn(output, target)
                loss.backward()
                self.optimizer.step()
                # Record for metrics
                train_loss += loss.item()

            train_acc, train_loss, _, _, _ = self._run_validation(train_input, num_classes)
            val_acc, val_loss, _, _, _ = self._run_validation(val_input, num_classes)

            self._record_metrics(
                verbose, i + 1, train_loss, val_loss, train_acc, val_acc
            )
            end = time.time()
            if verbose:
                print(f"Time Elapsed: {end - start:.2f}s")
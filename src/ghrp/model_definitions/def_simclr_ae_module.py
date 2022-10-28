import torch
import torch.nn as nn
import numpy as np
from .components.def_AE import AE, AE_attn, AE_perceiver
from .components.def_loss import GammaContrastReconLoss
from ghrp.checkpoints_to_datasets.dataset_auxiliaries import printProgressBar
from torch.utils.tensorboard import SummaryWriter
import itertools


class SimCLRAEModule(nn.Module):
    def __init__(self, config):
        super(SimCLRAEModule, self).__init__()

        self.verbosity = config.get("verbosity", 0)

        if self.verbosity > 0:
            print("Initialize Model")

        self.device = config.get("device", torch.device("cpu"))
        if type(self.device) is not torch.device:
            self.device = torch.device(self.device)
        if self.verbosity > 0:
            print(f"device: {self.device}")

        # setting seeds for reproducibility
        # https://pytorch.org/docs/stable/notes/randomness.html
        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        # if not CPU -> GPU: set cuda seeds
        if self.device is not torch.device("cpu"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.type = config.get("model::type", "vanilla")
        if self.type == "vanilla":
            model = AE(config)
        elif self.type == "transformer":
            model = AE_attn(config)
        elif self.type == "perceiver":
            model = AE_perceiver(config)

        self.model = model
        projection_head = (
            True if config.get("model::projection_head_layers", None) > 0 else False
        )

        #
        self.criterion = GammaContrastReconLoss(
            gamma=config.get("training::gamma", 0.5),
            reduction=config.get("training::reduction", "global_mean"),
            batch_size=config.get("trainset::batchsize", 64),
            temperature=config.get("training::temperature", 0.1),
            device=self.device,
            contrast=config.get("training::contrast", "simclr"),
            projection_head=projection_head,
            threshold=config.get("training::error_threshold", None),
            z_var_penalty=config.get("training::z_var_penalty", 0.0),
            config=config,
        )
        # send model and criterion to device
        self.model.to(self.device)
        self.criterion.to(self.device)

        # initialize model in eval mode
        self.model.eval()

        # gather model parameters and projection head parameters
        # params_lst = [self.model.parameters(), self.criterion.parameters()]
        # self.params = itertools.chain(*params_lst)
        self.params = self.parameters()

        # set optimizer
        self.set_optimizer(config)

        # half precision
        self.use_half = (
            True if config.get("training::precision", "full") == "half" else False
        )
        if self.use_half:
            print(f"++++++ USE HALF PRECISION +++++++")
            self.model = self.model.half()
            self.criterion = self.criterion.half()

        # automatic mixed precision
        self.use_amp = (
            True if config.get("training::precision", "full") == "amp" else False
        )
        if self.use_amp:
            print(f"++++++ USE AUTOMATIC MIXED PRECISION +++++++")
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # set trackers
        self.best_epoch = None
        self.loss_best = None
        self.best_checkpoint = None

        # mean loss for r^2
        self.loss_mean = None

        # init scheduler
        self.set_scheduler(config)

        self._save_model_checkpoint = True

        self.feature_normalization_koeff = None

        # init gradien clipping
        if config.get("training::gradient_clipping", None) == "norm":
            self.clip_grads = self.clip_grad_norm
            self.clipping_value = config.get("training::gradient_clipp_value", 5)
        elif config.get("training::gradient_clipping", None) == "value":
            self.clip_grads = self.clip_grad_value
            self.clipping_value = config.get("training::gradient_clipp_value", 5)
        else:
            self.clip_grads = None

    def set_feature_normalization(self, reference_weights, index_dict):
        # compute std of the weights __per layer__
        norm_std = []
        for start, length in zip(index_dict["idx_start"], index_dict["idx_length"]):
            idx_start = start
            idx_end = start + length
            sliced = reference_weights[:, idx_start:idx_end]
            tmp = sliced.flatten()
            std_tmp = torch.std(tmp)
            # apply thresholding to prevent division by zero
            epsilon = 1e-4
            if std_tmp.item() < epsilon:
                std_tmp = torch.tensor(1) * epsilon
            std = torch.ones(sliced.shape[1]) * std_tmp
            norm_std.append(std)
        norm_std = torch.cat(norm_std, dim=0)
        # set norm in recon loss
        self.feature_normalization_koeff = norm_std

    def clip_grad_norm(
        self,
    ):
        # print(f"clip grads by norm")
        # nn.utils.clip_grad_norm_(self.params, self.clipping_value)
        nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

    def clip_grad_value(
        self,
    ):
        # print(f"clip grads by value")
        # nn.utils.clip_grad_value_(self.params, self.clipping_value)
        nn.utils.clip_grad_value_(self.parameters(), self.clipping_value)

    def forward(self, x):
        # pass forward call through to model
        z = self.forward_encoder(x)
        y = self.forward_decoder(z)
        return z, y

    def forward_encoder(self, x):
        # normalize input features
        if self.feature_normalization_koeff is not None:
            dev = x.device
            self.feature_normalization_koeff = self.feature_normalization_koeff.to(dev)
            x /= self.feature_normalization_koeff
        z = self.model.forward_encoder(x)
        return z

    def forward_decoder(self, z):
        y = self.model.forward_decoder(z)
        # map output features back to original feature space
        if self.feature_normalization_koeff is not None:
            dev = y.device
            self.feature_normalization_koeff = self.feature_normalization_koeff.to(dev)
            y *= self.feature_normalization_koeff
        return y

    def forward_embeddings(self, x):
        z = self.forward_encoder(x)
        return z

    def set_optimizer(self, config):
        if config.get("optim::optimizer", "adamw") == "sgd":
            self.optimizer = torch.optim.SGD(
                # params=self.params,
                params=self.parameters(),
                lr=config.get("optim::lr", 3e-4),
                momentum=config.get("optim::momentum", 0.9),
                weight_decay=config.get("optim::wd", 3e-5),
            )
        elif config.get("optim::optimizer", "adamw") == "adam":
            self.optimizer = torch.optim.Adam(
                # params=self.params,
                params=self.parameters(),
                lr=config.get("optim::lr", 3e-4),
                weight_decay=config.get("optim::wd", 3e-5),
            )
        elif config.get("optim::optimizer", "adamw") == "adamw":
            self.optimizer = torch.optim.AdamW(
                # params=self.params,
                params=self.parameters(),
                lr=config.get("optim::lr", 3e-4),
                weight_decay=config.get("optim::wd", 3e-5),
            )
        elif config.get("optim::optimizer", "adamw") == "lamb":
            self.optimizer = torch.optim.Lamb(
                # params=self.params,
                params=self.parameters(),
                lr=config.get("optim::lr", 3e-4),
                weight_decay=config.get("optim::wd", 3e-5),
            )
        else:
            raise NotImplementedError(
                f'the optimizer {config.get("optim::optimizer", "adam")} is not implemented. break'
            )

    def set_scheduler(self, config):
        if config.get("optim::scheduler", None) == None:
            self.scheduler = None
        elif config.get("optim::scheduler", None) == "ReduceLROnPlateau":
            mode = config.get("optim::scheduler_mode", "min")
            factor = config.get("optim::scheduler_factor", 0.1)
            patience = config.get("optim::scheduler_patience", 10)
            threshold = config.get("optim::scheduler_threshold", 1e-4)
            threshold_mode = config.get("optim::scheduler_threshold_mode", "rel")
            cooldown = config.get("optim::scheduler_cooldown", 0)
            min_lr = config.get("optim::scheduler_min_lr", 0.0)
            eps = config.get("optim::scheduler_eps", 1e-8)

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold,
                threshold_mode=threshold_mode,
                cooldown=cooldown,
                min_lr=min_lr,
                eps=eps,
                verbose=False,
            )

    def set_normalization(self, reference_weights, index_dict):
        self.criterion.loss_recon.set_normalization(reference_weights, index_dict)

    def save_model(self, epoch, perf_dict, path=None):
        if path is not None:
            fname = path.joinpath(f"model_epoch_{epoch}.ptf")
            if self._save_model_checkpoint:
                # save model state-dict
                perf_dict["state_dict"] = self.model.state_dict()
                if self.criterion.loss_contrast is not None:
                    perf_dict[
                        "projection_head"
                    ] = self.criterion.loss_contrast.projection_head.state_dict()
                # save optimizer state-dict
                perf_dict["optimizer_state"] = self.optimizer.state_dict()
            torch.save(perf_dict, fname)
        return None

    # ##########################
    # one training step / batch
    # ##########################
    def train_step(self, x_i, x_j):
        # zero grads before training steps
        self.optimizer.zero_grad()
        if self.use_half:
            x_i, x_j = x_i.half(), x_j.half()
        # forward pass with both views
        z_i, y_i = self.forward(x_i)
        z_j, y_j = self.forward(x_j)
        # cat y_i, y_j and x_i, x_j
        x = torch.cat([x_i, x_j], dim=0)
        y = torch.cat([y_i, y_j], dim=0)
        # compute loss
        perf = self.criterion(z_i=z_i, z_j=z_j, y=y, t=x)
        # prop loss backwards to
        loss = perf["loss"]
        loss.backward()
        # gradient clipping
        if self.clip_grads is not None:
            self.clip_grads()
        # update parameters
        self.optimizer.step()
        # for key in perf.keys():
        #     perf[key] = perf[key].item()
        # compute embedding properties
        z_norm = torch.linalg.norm(z_i, ord=2, dim=1).mean()
        z_var = torch.mean(torch.var(z_i, dim=0))
        perf["z_norm"] = z_norm
        perf["z_var"] = z_var
        return perf

    # ##########################
    # one training step / batch
    # ##########################
    def train_step_amp(self, x_i, x_j):
        with torch.cuda.amp.autocast(enabled=True):
            # forward pass with both views
            z_i, y_i = self.forward(x_i)
            z_j, y_j = self.forward(x_j)
            # cat y_i, y_j and x_i, x_j
            x = torch.cat([x_i, x_j], dim=0)
            y = torch.cat([y_i, y_j], dim=0)
            # compute loss
            perf = self.criterion(z_i=z_i, z_j=z_j, y=y, t=x)
            # prop loss backwards to
            loss = perf["loss"]
        # backward
        # technically, there'd need to be a scaler for each loss individually.
        self.scaler.scale(loss).backward()
        # if gradient clipping is to be used...
        if self.clip_grads is not None:
            # # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)
            # Since the gradients of optimizer's assigned params are now unscaled, clips as usual.
            self.clip_grads()
        # update parameters
        self.scaler.step(self.optimizer)
        # update scaler
        self.scaler.update()
        # zero grads
        self.optimizer.zero_grad()
        # for key in perf.keys():
        #     perf[key] = perf[key].item()
        # compute embedding properties
        z_norm = torch.linalg.norm(z_i, ord=2, dim=1).mean()
        z_var = torch.mean(torch.var(z_i, dim=0))
        perf["z_norm"] = z_norm
        perf["z_var"] = z_var
        return perf

    # one training epoch
    def train_epoch(self, trainloader, epoch, writer=None, tf_out=10):
        if self.verbosity > 2:
            print(f"train epoch {epoch}")
        # set model to training mode
        self.model.train()
        self.criterion.train()

        if self.verbosity > 2:
            printProgressBar(
                0,
                len(trainloader),
                prefix="Batch Progress:",
                suffix="Complete",
                length=50,
            )
        # init accumulated loss, accuracy
        perf_out = {}
        n_data = 0
        # enter loop over batches
        for idx, data in enumerate(trainloader):
            x_i, l_i, x_j, _ = data
            # send to device
            x_i = x_i.to(self.device)
            x_j = x_j.to(self.device)  # take one training step

            if self.verbosity > 2:
                printProgressBar(
                    idx + 1,
                    len(trainloader),
                    prefix="Batch Progress:",
                    suffix="Complete",
                    length=50,
                )
            # compute loss
            if self.use_amp:
                perf = self.train_step_amp(x_i, x_j)
            else:
                perf = self.train_step(x_i, x_j)
            # scale loss with batchsize (get's normalized later)
            for key in perf.keys():
                if key not in perf_out:
                    perf_out[key] = perf[key] * len(l_i)
                else:
                    perf_out[key] += perf[key] * len(l_i)
            n_data += len(l_i)

        self.model.eval()
        self.criterion.eval()
        # compute epoch running losses
        for key in perf_out.keys():
            perf_out[key] /= n_data
            perf_out[key] = perf_out[key].item()
        # rsq_running = 1 - loss_running_recon / self.loss_mean

        # scheduler
        if self.scheduler is not None:
            self.scheduler.step(perf_out["loss"])

        return perf_out

    # test batch
    def test_step(self, x_i, x_j):
        with torch.no_grad():
            if self.use_half:
                x_i, x_j = x_i.half(), x_j.half()
            # forward pass with both views
            z_i, y_i = self.forward(x_i)
            z_j, y_j = self.forward(x_j)
            # cat y_i, y_j and x_i, x_j
            x = torch.cat([x_i, x_j], dim=0)
            y = torch.cat([y_i, y_j], dim=0)
            # compute loss
            perf = self.criterion(z_i=z_i, z_j=z_j, y=y, t=x)
            # for key in perf.keys():
            #     perf[key] = perf[key].item()
            return perf

    # test batch
    def test_step_amp(self, x_i, x_j):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                # forward pass with both views
                z_i, y_i = self.forward(x_i)
                z_j, y_j = self.forward(x_j)
                # cat y_i, y_j and x_i, x_j
                x = torch.cat([x_i, x_j], dim=0)
                y = torch.cat([y_i, y_j], dim=0)
                # compute loss
                perf = self.criterion(z_i=z_i, z_j=z_j, y=y, t=x)
            # for key in perf.keys():
            #     perf[key] = perf[key].item()
            return perf

    # test epoch
    def test_epoch(self, testloader, epoch, writer=None, tf_out=10):
        if self.verbosity > 2:
            print(f"test at epoch {epoch}")
        # set model to eval mode
        self.model.eval()
        self.criterion.eval()
        # init accumulated loss, accuracy
        perf_out = {}
        n_data = 0
        # enter loop over batches
        for idx, data in enumerate(testloader):
            x_i, l_i, x_j, _ = data
            # send to device
            x_i = x_i.to(self.device)
            x_j = x_j.to(self.device)  # take one training step
            # compute loss
            if self.use_amp:
                perf = self.test_step_amp(x_i, x_j)
            else:
                perf = self.test_step(x_i, x_j)
            # scale loss with batchsize (get's normalized later)
            for key in perf.keys():
                if key not in perf_out:
                    perf_out[key] = perf[key] * len(l_i)
                else:
                    perf_out[key] += perf[key] * len(l_i)
            n_data += len(l_i)

        # compute epoch running losses
        for key in perf_out.keys():
            perf_out[key] /= n_data
            perf_out[key] = perf_out[key].item()

        return perf_out

    # training loop over all epochs
    def train_loop(self, config):
        if self.verbosity > 0:
            print("##### enter training loop ####")

        # unpack training_config
        epochs_train = config["training::epochs_train"]
        start_epoch = config["training::start_epoch"]
        output_epoch = config["training::output_epoch"]
        test_epochs = config["training::test_epochs"]
        tf_out = config["training::tf_out"]
        checkpoint_dir = config["training::checkpoint_dir"]
        tensorboard_dir = config["training::tensorboard_dir"]

        if tensorboard_dir is not None:
            tb_writer = SummaryWriter(log_dir=tensorboard_dir)
        else:
            tb_writer = None

        # trainloaders with matching lenghts

        trainloader = config["training::trainloader"]
        testloader = config["training::testloader"]

        ## compute loss_mean
        self.loss_mean = self.criterion.compute_mean_loss(testloader)

        # compute initial test loss
        loss_test, loss_test_contr, loss_test_recon, rsq_test = self.test(
            testloader,
            epoch=0,
            writer=tb_writer,
            tf_out=tf_out,
        )

        # write first state_dict
        perf_dict = {
            "loss_train": 1e15,
            "loss_test": loss_test,
            "rsq_train": -999,
            "rsq_test": rsq_test,
        }

        self.save_model(epoch=0, perf_dict=perf_dict, path=checkpoint_dir)
        self.best_epoch = 0
        self.loss_best = 1e15

        # initialize the epochs list
        epoch_iter = range(start_epoch, start_epoch + epochs_train)
        # enter training loop
        # for epoch in epoch_iter:

        #     # enter training loop over all batches
        #     loss_train, loss_train_contr, loss_train_recon, rsq_train = self.train(
        #         trainloader, epoch, writer=tb_writer, tf_out=tf_out
        #     )

        #     if epoch % test_epochs == 0:
        #         perf_test = self.test(
        #             testloader,
        #             epoch,
        #             writer=tb_writer,
        #             tf_out=tf_out,
        #         )

        #         if loss_test < self.loss_best:
        #             self.best_epoch = epoch
        #             self.loss_best = loss_test
        #             self.best_checkpoint = self.model.state_dict()
        #             perf_dict["epoch"] = epoch
        #             perf_dict["loss_train"] = loss_train
        #             perf_dict["loss_train_contr"] = loss_train_contr
        #             perf_dict["loss_train_recon"] = loss_train_recon
        #             perf_dict["rsq_train"] = rsq_train
        #             perf_dict["loss_test"] = loss_test
        #             perf_dict["loss_test_contr"] = loss_test_contr
        #             perf_dict["loss_test_recon"] = loss_test_recon
        #             perf_dict["rsq_test"] = rsq_test
        #             if checkpoint_dir is not None:
        #                 self.save_model(
        #                     epoch="best", perf_dict=perf_dict, path=checkpoint_dir
        #                 )
        #         # if self.verbosity > 1:
        #         # print(f"best loss: {self.loss_best} at epoch {self.best_epoch}")

        #     if epoch % output_epoch == 0:
        #         perf_dict["epoch"] = epoch
        #         perf_dict["loss_train"] = loss_train
        #         perf_dict["loss_train_contr"] = loss_train_contr
        #         perf_dict["loss_train_recon"] = loss_train_recon
        #         perf_dict["rsq_train"] = rsq_train
        #         perf_dict["loss_test"] = loss_test
        #         perf_dict["loss_test_contr"] = loss_test_contr
        #         perf_dict["loss_test_recon"] = loss_test_recon
        #         perf_dict["rsq_test"] = rsq_test
        #         if checkpoint_dir is not None:
        #             self.save_model(
        #                 epoch=epoch, perf_dict=perf_dict, path=checkpoint_dir
        #             )
        #         if self.verbosity > 1:
        #             print(
        #                 f"epoch {epoch}:: train_loss = {loss_train}; train r**2 {rsq_train}; loss_train_contr: {loss_train_contr}; loss_train_recon: {loss_train_recon}"
        #             )
        #             print(
        #                 f"epoch {epoch}:: test_loss = {loss_test}; test r**2 {rsq_test}; loss_test_contr: {loss_test_contr}; loss_test_recon: {loss_test_recon}"
        #             )

        self.last_checkpoint = self.model.state_dict()
        return self.loss_best

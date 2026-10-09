import math
import torch
try:
    import torch_npu
except ImportError:
    pass
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from utils.state_dict_ops import load_param_dict_, model_param_dict, sd_lerp

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


class Local(object):
    """
    Local 父类：
    1. 统一管理 dataloader、batch 处理、forward、train loop
    2. 子类只关注优化器初始化和梯度修正策略
    """

    def __init__(
        self,
        args,
        dataset=None,
        idxs=None,
        iters=None,
        nums=None,
        momentum_beta=None,
        h=None,
        local_c=None,
        g_mix=None,
        rho=0,
        server_c=None,
    ):
        self.args = args
        self.iters = iters
        self.nums = nums

        self.momentum_beta = momentum_beta
        self.h = h
        self.local_c = local_c
        self.g_mix = g_mix
        self.server_c = server_c
        self.rho = rho

        self.loss_func = nn.CrossEntropyLoss()
        self.last_ec = {}

        batch_size = args.local_bs

        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs),
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        local_epochs = getattr(args, 'local_epochs', None)
        if local_epochs is not None and float(local_epochs) > 0:
            self.iters = max(1, int(math.ceil(len(self.ldr_train) * float(local_epochs))))

    @property
    def device(self):
        return self.args.device

    def _named_trainable_params(self, net):
        for name, param in net.named_parameters():
            if param.requires_grad:
                yield name, param

    def _init_reference_params(self, net):
        if self.rho <= 0:
            return None
        init_params = {}
        for name, param in self._named_trainable_params(net):
            init_params[name] = param.data.clone().detach()
        return init_params

    def _process_batch(self, batch):
        if isinstance(batch, dict):
            return {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

        images, labels = batch
        return {
            'images': images.to(self.device),
            'labels': labels.to(self.device)
        }

    def _forward_pass(self, net, inputs):
        if 'input_ids' in inputs:
            forward_kwargs = {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs.get('attention_mask'),
                'labels': inputs.get('labels'),
            }
            if 'token_type_ids' in inputs:
                forward_kwargs['token_type_ids'] = inputs['token_type_ids']

            forward_kwargs = {
                key: value for key, value in forward_kwargs.items() if value is not None
            }
            outputs = net(**forward_kwargs)
            if getattr(outputs, 'loss', None) is None and 'labels' in inputs:
                loss = self.loss_func(outputs.logits, inputs['labels'])
                return type('Output', (object,), {'loss': loss, 'logits': outputs.logits})()
            return outputs

        logits = net(inputs['images'])
        loss = self.loss_func(logits, inputs['labels'])
        return type('Output', (object,), {'loss': loss, 'logits': logits})()

    def _compute_loss_and_backward(self, net, outputs, init_params=None):
        loss = outputs.loss
        if self.rho > 0 and init_params is not None:
            reg_loss = 0.0
            for name, param in self._named_trainable_params(net):
                diff = param - init_params[name]
                reg_loss += torch.sum(diff * diff)
            total_loss = loss + 0.5 * self.rho * reg_loss
            total_loss.backward()
        else:
            loss.backward()
        return loss

    def build_optimizer(self, net):
        weight_decay = getattr(self.args, 'weight_decay', 0.0) or 0.0
        params = [param for param in net.parameters() if param.requires_grad]
        if not params:
            raise ValueError("No trainable parameters found for local training.")
        return torch.optim.SGD(
            params,
            lr=self.args.lr,
            weight_decay=weight_decay
        )

    def build_scheduler(self, optimizer):
        return optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[lambda epoch: 1.0]
        )

    def before_train(self, net, optimizer):
        pass

    def modify_gradients(self, net):
        raise NotImplementedError

    def after_step(self, net, step_idx):
        pass

    def after_train(self, net, steps_done):
        pass

    def average_stochastic_gradients(self, net):
        net.train()

        num_batch = len(self.ldr_train)
        if num_batch == 0 or self.iters is None or self.iters <= 0:
            return {}

        avg_grads = {}
        for name, param in self._named_trainable_params(net):
            avg_grads[name] = torch.zeros_like(param.data, device=self.device)

        steps_done = 0
        data_iter = iter(self.ldr_train)
        while steps_done < self.iters:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.ldr_train)
                batch = next(data_iter)

            net.zero_grad()
            inputs = self._process_batch(batch)
            outputs = self._forward_pass(net, inputs)
            self._compute_loss_and_backward(net, outputs, init_params=None)

            for name, param in self._named_trainable_params(net):
                if param.grad is not None:
                    avg_grads[name] += param.grad.detach()

            steps_done += 1

        for name in avg_grads.keys():
            avg_grads[name] /= steps_done

        return avg_grads

    def train(self, net):
        net.train()

        num_batch = len(self.ldr_train)
        if num_batch == 0 or self.iters is None or self.iters <= 0:
            return model_param_dict(net, device=self.device)

        init_params = self._init_reference_params(net)
        optimizer = self.build_optimizer(net)
        scheduler = self.build_scheduler(optimizer)

        self.before_train(net, optimizer)

        steps_done = 0
        while steps_done < self.iters:
            for batch in self.ldr_train:
                if steps_done >= self.iters:
                    break

                inputs = self._process_batch(batch)
                net.zero_grad()

                outputs = self._forward_pass(net, inputs)
                self._compute_loss_and_backward(net, outputs, init_params)
                self.modify_gradients(net)

                optimizer.step()
                scheduler.step()

                steps_done += 1
                self.after_step(net, steps_done)

        self.after_train(net, steps_done)
        return model_param_dict(net, device=self.device)


class LocalSGD(Local):
    """普通本地 SGD。"""

    def modify_gradients(self, net):
        return













class LocalAdaMasFL(Local):
    """
    AdaMasFL client recurrence from Algorithm 4.

    The stale global control and momentum are kept separately so the client
    computes beta*(grad - local_c + stale_c) + (1-beta)*stale_g.
    """

    def before_train(self, net, optimizer):
        self.g_next = self._init_momentum_buffer(net)
        self.sum_grads = {}
        for name, param in self._named_trainable_params(net):
            self.sum_grads[name] = torch.zeros_like(param.data, device=self.device)

    def _init_momentum_buffer(self, net):
        momentum_buffer = {}
        for name, param in self._named_trainable_params(net):
            if self.h is not None and name in self.h:
                momentum_buffer[name] = self.h[name].detach().clone().to(self.device)
            else:
                momentum_buffer[name] = torch.zeros_like(param.data, device=self.device)
        return momentum_buffer

    def modify_gradients(self, net):
        sqsum = 0.0

        for name, param in self._named_trainable_params(net):
            if param.grad is None:
                continue

            local_c = getattr(self, 'local_c', None)
            server_c = getattr(self, 'server_c', None)
            g_mix = getattr(self, 'g_mix', None)
            local_c_value = local_c[name] if local_c is not None and name in local_c else torch.zeros_like(param.grad)
            server_c_value = server_c[name] if server_c is not None and name in server_c else torch.zeros_like(param.grad)
            g_mix_value = g_mix[name] if g_mix is not None and name in g_mix else torch.zeros_like(param.grad)

            self.g_next[name] = self.momentum_beta * (param.grad - local_c_value + server_c_value) + (1.0 - self.momentum_beta) * g_mix_value
            self.sum_grads[name] += param.grad.detach()
            sqsum += torch.sum(self.g_next[name] ** 2)

        norm = torch.sqrt(sqsum).clamp_min(torch.finfo(sqsum.dtype).tiny)

        for name, param in self._named_trainable_params(net):
            if param.grad is None:
                continue
            param.grad = self.g_next[name] / norm

    def after_train(self, net, steps_done):
        if steps_done <= 0:
            return
        for name, grad_sum in self.sum_grads.items():
            self.last_ec[name] = grad_sum / steps_done

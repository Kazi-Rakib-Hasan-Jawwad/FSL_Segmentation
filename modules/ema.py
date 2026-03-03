import torch
import torch.nn as nn

class EMAWarmUp(nn.Module):
    def __init__(self, model, gamma=0.999, warm_up_steps=100):
        super().__init__()
        
        self.model_name_to_shadow_name = {}
        self.register_buffer('gamma', torch.tensor(gamma, dtype=torch.float32))
        self.register_buffer('warm_up_step', torch.tensor(1 if warm_up_steps > 0 else -1, dtype=torch.int))
        self.register_buffer('warm_up_steps', torch.tensor(warm_up_steps, dtype=torch.int))

        for name, params in model.named_parameters():
            if params.requires_grad:
                shadow_module_name = name.replace('.', '')
                self.model_name_to_shadow_name.update({ name : shadow_module_name })
                self.register_buffer(shadow_module_name, params.clone().detach().data)

        self.collected_params = []

    def forward(self, model):
        gamma = self.gamma

        if self.warm_up_step > 0:
            gamma = self.gamma if self.warm_up_step >= self.warm_up_steps else self.gamma * self.warm_up_step / self.warm_up_steps
            self.warm_up_step += 1

        with torch.no_grad():
            model_params = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key, model_param in model_params.items():
                if model_param.requires_grad:
                    shadow_name = self.model_name_to_shadow_name[key]
                    shadow_params[shadow_name] = shadow_params[shadow_name].type_as(model_param)
                    shadow_params[shadow_name].sub_((1-gamma) * (shadow_params[shadow_name] - model_param))
                else:
                    assert not key in self.model_name_to_shadow_name

    def copy_to(self, model):
        model_params = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key, model_param in model_params.items():
            if model_param.requires_grad:
                model_param.data.copy_(shadow_params[self.model_name_to_shadow_name[key]].data)
            else:
                assert not key in self.model_name_to_shadow_name

    def store(self, parameters):
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        for collected_param, param in zip(self.collected_params, parameters):
            param.data.copy_(collected_param.data)

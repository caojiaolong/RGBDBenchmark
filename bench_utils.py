import sys
import torch
from calflops import calculate_flops


def cal_flops_params(
    model, shape=None, dataset="nyudv2", single_channel=False, cuda=False, batch_size=1
):
    """
    calculate the flops and params of the model
    :param model: the model to calculate
    :param input_shape: the input shape of the model, e.g. (480, 640)
    :param dataset: the dataset used to train the model
    :param single_channel: whether the model uses single channel modal x input
    :param cuda: whether to use cuda
    """
    if shape is None:
        if dataset == "nyudv2":
            shape = (480, 640)
        elif dataset == "sunrgbd":
            shape = (530, 730)
        else:
            raise NotImplementedError
    input_shape = (batch_size, 3, *shape)
    input_x_shape = (
        (batch_size, 1, *shape) if single_channel else (batch_size, 3, *shape)
    )
    input = torch.randn(input_shape).cuda() if cuda else torch.randn(input_shape)
    input_x = torch.randn(input_x_shape).cuda() if cuda else torch.randn(input_x_shape)
    model = model.cuda() if cuda else model
    flops, macs, params = calculate_flops(
        model=model,
        args=[input, input_x],
        print_results=False,
        output_precision=4,
    )
    return (flops, macs, params)


def decorate_model_for_zip_input(model):
    """
    decorate the model for zip input

    :param model: the model to decorate
    """

    class ZipInputModel(torch.nn.Module):
        def __init__(self, model):
            super(ZipInputModel, self).__init__()
            self.model = model

        def forward(self, rgb, x):
            return self.model([rgb, x])

    return ZipInputModel(model)


def decorate_model_for_dict_input(model, input_keys: list):
    """
    decorate the model for dict input

    :param model: the model to decorate
    :param input_keys: the input keys
    """

    class ZipInputModel(torch.nn.Module):
        def __init__(self, model):
            super(ZipInputModel, self).__init__()
            self.model = model

        def forward(self, rgb, x):
            return self.model({input_keys[0]: rgb, input_keys[1]: x})

    return ZipInputModel(model)


def decorate_model_for_stack_input(model):
    """
    decorate the model for stack input

    :param model: the model to decorate
    """

    class ZipInputModel(torch.nn.Module):
        def __init__(self, model):
            super(ZipInputModel, self).__init__()
            self.model = model

        def forward(self, rgb, x):
            rgbd = torch.cat([rgb, x], dim=1)
            return self.model(rgbd)

    return ZipInputModel(model)


def decorate_model_for_omnivore(model):
    """
    decorate the model for dict input

    :param model: the model to decorate
    :param input_keys: the input keys
    """

    class omnivoreModel(torch.nn.Module):
        def __init__(self, model):
            super(omnivoreModel, self).__init__()
            self.model = model

        def forward(self, rgb, x):
            rgbd = torch.cat([rgb, x], dim=1)
            rgbd_input = rgbd[
                :, :, None, ...
            ]  # The model expects inputs of shape: B x C x T x H x W
            return self.model(rgbd_input, input_type="rgbd")

    return omnivoreModel(model)


class Envs_state_manager:
    def __init__(self):
        self.init_sys_modules = set(sys.modules.keys())
        self.sys_path = sys.path.copy()
        print("Envs_state_manager initialized, sys.modules and sys.path saved")

    def restore(self):
        current_sys_modules = sys.modules.keys()
        new_sys_modules = set(current_sys_modules) - self.init_sys_modules
        for module in new_sys_modules:
            if "cv2" in module:
                # for some reason, deleting cv2 module will cause error
                continue
            else:
                del sys.modules[module]
        print(f"Deleted module: {new_sys_modules}")
        print("Envs_state_manager restored, new modules deleted! ")
        sys.path = self.sys_path.copy()
        torch.cuda.empty_cache()

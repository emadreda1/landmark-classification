# import torch
# import torch.nn as nn
# import torch.optim


# def get_loss():
#     return nn.CrossEntropyLoss()


# def get_optimizer(
#     model: nn.Module,
#     optimizer: str = "SGD",
#     learning_rate: float = 0.01,
#     momentum: float = 0.5,
#     weight_decay: float = 0,
# ):
#     if optimizer.lower() == "sgd":
#         return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
#     elif optimizer.lower() == "adam":
#         return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     else:
#         raise ValueError(f"Optimizer {optimizer} not supported")


# ######################################################################################
# #                                     TESTS
# ######################################################################################
# import pytest


# @pytest.fixture(scope="session")
# def fake_model():
#     return nn.Linear(16, 256)


# def test_get_loss():
#     loss = get_loss()
#     assert isinstance(loss, nn.CrossEntropyLoss), f"Expected cross entropy loss, found {type(loss)}"


# def test_get_optimizer_type(fake_model):
#     opt = get_optimizer(fake_model)
#     assert isinstance(opt, torch.optim.SGD), f"Expected SGD optimizer, got {type(opt)}"


# def test_get_optimizer_is_linked_with_model(fake_model):
#     opt = get_optimizer(fake_model)
#     assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])


# def test_get_optimizer_returns_adam(fake_model):
#     opt = get_optimizer(fake_model, optimizer="adam")
#     assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])
#     assert isinstance(opt, torch.optim.Adam), f"Expected SGD optimizer, got {type(opt)}"


# def test_get_optimizer_sets_learning_rate(fake_model):
#     opt = get_optimizer(fake_model, optimizer="adam", learning_rate=0.123)
#     assert opt.param_groups[0]["lr"] == 0.123, "get_optimizer is not setting the learning rate appropriately. Check your code."


# def test_get_optimizer_sets_momentum(fake_model):
#     opt = get_optimizer(fake_model, optimizer="SGD", momentum=0.123)
#     assert opt.param_groups[0]["momentum"] == 0.123, "get_optimizer is not setting the momentum appropriately. Check your code."


# def test_get_optimizer_sets_weight_decat(fake_model):
#     opt = get_optimizer(fake_model, optimizer="SGD", weight_decay=0.123)
#     assert opt.param_groups[0]["weight_decay"] == 0.123, "get_optimizer is not setting the weight_decay appropriately. Check your code."

# optimization.py
import torch
import torch.nn as nn
import torch.optim


# def get_loss():
#     return nn.CrossEntropyLoss()


# def get_optimizer(
#     model: nn.Module,
#     optimizer: str = "Adam",
#     learning_rate: float = 0.001,
#     momentum: float = 0.9,
#     weight_decay: float = 0.0001,
# ):
#     if optimizer.lower() == "sgd":
#         return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
#     elif optimizer.lower() == "adam":
#         return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     else:
#         raise ValueError(f"Optimizer {optimizer} not supported")

import torch.optim as optim
import torch.nn as nn

def get_optimizer(model, optimizer, learning_rate, weight_decay):
    if optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown optimizer")

def get_loss():
    return nn.CrossEntropyLoss()

######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def fake_model():
    return nn.Linear(16, 256)


def test_get_loss():
    loss = get_loss()
    assert isinstance(loss, nn.CrossEntropyLoss), f"Expected cross entropy loss, found {type(loss)}"


def test_get_optimizer_type(fake_model):
    opt = get_optimizer(fake_model)
    assert isinstance(opt, torch.optim.Adam), f"Expected Adam optimizer, got {type(opt)}"


def test_get_optimizer_is_linked_with_model(fake_model):
    opt = get_optimizer(fake_model)
    assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])


def test_get_optimizer_returns_sgd(fake_model):
    opt = get_optimizer(fake_model, optimizer="sgd")
    assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])
    assert isinstance(opt, torch.optim.SGD), f"Expected SGD optimizer, got {type(opt)}"


def test_get_optimizer_sets_learning_rate(fake_model):
    opt = get_optimizer(fake_model, optimizer="adam", learning_rate=0.001)
    assert opt.param_groups[0]["lr"] == 0.001, "get_optimizer is not setting the learning rate appropriately. Check your code."


def test_get_optimizer_sets_weight_decay(fake_model):
    opt = get_optimizer(fake_model, optimizer="adam", weight_decay=0.0001)
    assert opt.param_groups[0]["weight_decay"] == 0.0001, "get_optimizer is not setting the weight decay appropriately. Check your code."

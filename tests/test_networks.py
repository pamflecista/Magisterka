import pytest
from bin.networks import *
import torch


#@pytest.mark.parametrize("shape", {[1, 1, 4, 10], [4, 1, 4, 100]})
def test_BassetNetwork(shape):
    x = torch.randint(0, 100, shape)
    network = BassetNetwork(shape[3])
    results = network(x.float())
    print(results)


test_BassetNetwork([1, 1, 4, 15])

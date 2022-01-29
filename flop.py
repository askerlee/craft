from fvcore.nn import FlopCountAnalysis
import torch

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x, y):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        #h_relu = self.linear1(x+y).clamp(min=0)
        #y_pred = self.linear2(h_relu)
        z = torch.matmul(x, y)
        return z

model = TwoLayerNet(10, 20, 5)
x = torch.randn(1, 4, 7040, 7040)
y = torch.randn(1, 4, 7040, 256)
flops=FlopCountAnalysis(model, (x, y))
print(flops.by_module())

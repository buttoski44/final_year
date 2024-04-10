export const code1 = {
  fileName: "Model.py",
  code: `
  from defines import *
  import numpy as np
  import mobula
  import mobula.layers as L
  
  class LeNet5:
      def __init__(self, X, labels):
  
          data, label = L.Data([X, labels], "data", batch_size = 100)
          conv1 = L.Conv(data, dim_out = 20, kernel = 5)
          pool1 = L.Pool(conv1, pool = L.Pool.MAX, kernel = 2, stride = 2)
          relu1 = L.ReLU(pool1)
          conv2 = L.Conv(relu1, dim_out = 50, kernel = 5)
          pool2 = L.Pool(conv2, pool = L.Pool.MAX, kernel = 2, stride = 2)
          relu2 = L.ReLU(pool2)
          fc3   = L.FC(relu2, dim_out = 500)
          relu3 = L.ReLU(fc3)
          pred  = L.FC(relu3, "pred", dim_out = 10)
          loss  = L.SoftmaxWithLoss(pred, "loss", label = label)
  
          # Net Instance
          self.net = mobula.Net()
  
          # Set Loss Layer
          self.net.set_loss(loss)
  
      @property
      def Y(self):
          return self.net["pred"].Y
  
      @property
      def loss(self):
          return self.net["loss"].loss
  
      @property
      def label(self):
          return self.net["data"](1).Y
    `,
};

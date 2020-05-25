"""
@author: Francisco Maestre. Modified version of Viet Nguyen
"""

import torch.nn as nn
import torch.nn.functional as F

'''
Clase personalizada del ActorCritic pensada para los juegos en cuestión
'''

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        ## http://www.diegocalvo.es/red-neuronal-convolucional/
        ## https://hackernoon.com/pytorch-y-su-funcionamiento-0p5j32hs
        ## Capas de redes convuncionales de 2d que nos van a ayudar a tratar las imágenes
        num_conv_layers = 4
        self.conv_layers = []
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        ## Red neuronal LSTM. Su tamaño debe ser ajustado a la salida de las capas de convolución
        self.lstm = nn.LSTMCell(32 * 6 * 6, 512)
        ## Transformación linear del Actor y del Crítico. El Actor tiene como salida un valor para cada acción posible mientras que el crítico sólo 1
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))
        return self.actor_linear(hx), self.critic_linear(hx), hx, cx




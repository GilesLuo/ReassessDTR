
import torch.nn as nn
import torch.nn.functional as F

import torch

class Entropy(nn.Module):
    def __init__(self, is_logits=True):
        """
        Initialize the entropy calculator.
        :param is_logits: A boolean indicating whether the input to the 'calculate' method will be logits or probabilities.
        """
        super().__init__()
        self.is_logits = is_logits

    def forward(self, outputs):
        """
        Calculate the entropy of the given outputs.
        :param outputs: The output from the neural network. Could be logits or probabilities.
        :return: Tensor representing the entropy of each instance in the output.
        """
        if self.is_logits:
            probabilities = F.softmax(outputs, dim=1)
        else:
            probabilities = outputs

        # Avoid log(0) by adding a small constant
        probabilities = torch.clamp(probabilities, min=1e-9)

        # Calculate entropy
        entropy = -torch.sum(probabilities * torch.log(probabilities), dim=1)
        return entropy.mean()



# By Shatha Jaradat
# KTH - Royal Institute of Technology
# 2018

# Some parts of the code that are related to finetunning models was taken from the following link:
# Then customised according to my needs
#https://gist.github.com/panovr/2977d9f26866b05583b0c40d88a315bf


import torch.nn as nn
import torch.nn.functional as function
import Models.Constants_DeepFashion.Helper as helper



# Base Archiectures - Static (No Dynamic Connections or Dynamic Layers)
# Single class Multi Labels  - categories and attributes
class FineTuneModel(nn.Module):

    def __init__(self, original_model, arch):
        super(FineTuneModel, self).__init__()

        # Support for multiple archiectures
        if arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.linear1 = nn.Linear(512, 512)
            self.generalCategoriesLinear = nn.Linear(512,helper.Num_category_classes, bias=True)
            self.generalAttributesLinear = nn.Linear(512,helper.Num_attribute_classes_general, bias=True)
            self.modelName = 'resnet'

        if arch.startswith('vgg16'):
            self.features = original_model.features
            self.linear1 = nn.Linear(25088, 4096)
            self.linear2 = nn.Linear(4096, 4096)
            self.generalCategoriesLinear = nn.Linear(4096,helper.Num_category_classes, bias=True)
            self.generalAttributesLinear = nn.Linear(4096,helper.Num_attribute_classes_general, bias=True)
            self.modelName = 'vgg16'

        else:
            raise("Finetuning not supported on this architecture yet")

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = True

    def forward(self, x):
        f = self.features(x)

        if self.modelName == 'resnet':
            f = f.view(f.size(0), -1)

            hrelu_linear1 = function.relu(self.linear1(f).clamp(min=0))
            hrelu_generalcategories = function.relu((self.generalCategoriesLinear(hrelu_linear1)))
            hrelu_generalattributes = function.relu((self.generalAttributesLinear(hrelu_linear1)))
            lstOutputs = []

            # Float Tensor of size 1 * 50
            lstOutputs.append(hrelu_generalcategories)
            # Float Tensor of size 1 * 1000
            lstOutputs.append(hrelu_generalattributes)

        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)

            hrelu_linear1 = function.relu(self.linear1(f).clamp(min=0))
            hrelu_linear2 = function.relu(self.linear2(hrelu_linear1))
            hrelu_generalcategories = function.relu((self.generalCategoriesLinear(hrelu_linear2)))
            hrelu_generalattributes = function.relu((self.generalAttributesLinear(hrelu_linear2)))
            lstOutputs = []

            # Float Tensor of size 1 * 50
            lstOutputs.append(hrelu_generalcategories)
            # Float Tensor of size 1 * 1000
            lstOutputs.append(hrelu_generalattributes)

        return lstOutputs

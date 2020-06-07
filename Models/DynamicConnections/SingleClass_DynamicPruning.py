# By Shatha Jaradat
# KTH - Royal Institute of Technology
# 2018

# Some parts of the code that are related to finetunning models was taken from the following link:
# Then customised according to my needs
#https://gist.github.com/panovr/2977d9f26866b05583b0c40d88a315bf


import torch
import torch.nn as nn
import torch.nn.functional as function
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
import Models.Constants_DeepFashion.Helper as helper

######## Dynamic Pruning for Single Class Scenarios ######
######## Was applied on Deep Fashion Dataseat ##########
######## Supports VGG16 and ResNet archiectures ########
class FineTuneModel(nn.Module):

    def __init__(self, original_model, arch):
        super(FineTuneModel, self).__init__()

        if arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.linear1 = nn.Linear(512, 512)
            #self.linear2 = nn.Linear(512, 512)

            self.generalCategoriesLinear = nn.Linear(512,helper.Num_category_classes)
            self.generalAttributesLinear = nn.Linear(512,helper.Num_attribute_classes_general)

            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.linear1 = nn.Linear(25088, 4096)
            self.linear2 = nn.Linear(4096, 4096)
            self.generalCategoriesLinear = nn.Linear(4096,helper.Num_category_classes)
            self.generalAttributesLinear = nn.Linear(4096,helper.Num_attribute_classes_general)

            self.modelName = 'vgg16'
        else :
            raise("Finetuning not supported on this architecture yet")

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = True

    def forward(self, x, categoryItem_Value, attributeItem_Value, attribute2_Value):
        f = self.features(x)

        categoryAttributesMappings = {}
        with open(helper.categoryAttributesMappingPath,"r") as file_mappings:
            for line in file_mappings:
                categoryAttributesMappings[int(line.split(",")[0])] = line.split(",")[1]

        # initial choice of category to be used to decide dynamically the weights of
        # connections of "attributes"
        cat = int(categoryItem_Value)
        attr = int(attributeItem_Value)
        attr2 = int(attribute2_Value)

        # output contains the 1 * 50 tensor for categories
        # and a 1*1000 tensor for attributes (dynamically changing weights)
        lstOutputs = []
        lstIndices = []
        indx = 0
        with open(categoryAttributesMappings[cat],"r") as file_attributes:
            for line in file_attributes:
                lstIndices.append(int(line)) #index of the attribute in the 1000 dimension
                indx += 1

        lstIndices = sorted(lstIndices, key=int)

        if self.modelName == 'resnet':
            f = f.view(f.size(0), -1)
            hrelu_linear2 = function.relu(self.linear1(f).clamp(min=0))
            #hrelu_linear2 = function.relu(self.linear2(hrelu_linear1))
            hrelu_generalcategories = function.relu((self.generalCategoriesLinear(hrelu_linear2)))

            # initializations for the correct scope of attributes based
            # on chosen category
            arr = np.empty(512)
            for i in range(0,512):
                arr[i] = helper.weight_intialisation

            for counter in range(0,helper.Num_attribute_classes_general):
                # weights of unrelated attributes can be zero
                # required gradient is set to false
                if counter not in lstIndices:
                    self.generalAttributesLinear.weight.data[counter] = torch.zeros(512)
                    self.generalAttributesLinear.weight.data[counter].requires_grad = False
                else:
                    if counter == attr or counter == attr2: # when a supporting attribute is sent
                        # weights of connections can be slightly increased
                        # to see the effect of supporting text values
                        self.generalAttributesLinear.weight.data[counter].requires_grad = True
                        arr_temp = np.empty(512)
                        for i in range(0,512):
                             arr_temp[i] = self.generalAttributesLinear.weight.data[counter][i]
                             arr_temp[i] += helper.weight_adjustment
                        self.generalAttributesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
                    else:
                        self.generalAttributesLinear.weight.data[counter].requires_grad = True
                        self.generalAttributesLinear.weight.data[counter] = torch.from_numpy(arr)

            hrelu_attributes = function.relu((self.generalAttributesLinear(hrelu_linear2)))

            lstOutputs.append(hrelu_generalcategories)
            lstOutputs.append(hrelu_attributes)

            return lstOutputs

        elif self.modelName == 'vgg16':
            if self.modelName == 'vgg16':
                f = f.view(f.size(0), -1)

            hrelu_linear1 = function.relu(self.linear1(f).clamp(min=0))
            hrelu_linear2 = function.relu(self.linear2(hrelu_linear1))
            hrelu_generalcategories = function.relu((self.generalCategoriesLinear(hrelu_linear2)))

            # initializations for the correct scope of attributes based
            # on chosen category
            arr_temp2 = np.empty(4096)
            for i in range(0,4096):
                arr_temp2[i] = helper.weight_intialisation

            for counter in range(0,helper.Num_attribute_classes_general):
                # weights of unrelated attributes can be zero
                # required gradient is set to false
                if counter not in lstIndices:
                    self.generalAttributesLinear.weight.data[counter] = torch.zeros(4096)
                    self.generalAttributesLinear.weight.data[counter].requires_grad = False
                else:
                    if counter == attr or counter == attr2: # when a supporting attribute is sent
                        # weights of connections can be slightly increased
                        # to see the effect of supporting text values
                        self.generalAttributesLinear.weight.data[counter].requires_grad = True
                        arr_temp = np.empty(4096)
                        for i in range(0,4096):
                             arr_temp[i] = self.generalAttributesLinear.weight.data[counter][i]
                             arr_temp[i] += helper.weight_adjustment
                        self.generalAttributesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
                    else:
                        self.generalAttributesLinear.weight.data[counter].requires_grad = True
                        self.generalAttributesLinear.weight.data[counter] = torch.from_numpy(arr_temp2)

        hrelu_attributes = function.relu((self.generalAttributesLinear(hrelu_linear2)))

        lstOutputs.append(hrelu_generalcategories)
        lstOutputs.append(hrelu_attributes)

        return lstOutputs
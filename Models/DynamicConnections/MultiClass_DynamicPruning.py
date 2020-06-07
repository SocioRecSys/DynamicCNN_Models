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
import Models.Constants_Instagram.Helper as helper


######## Dynamic Pruning for Multi Class Scenarios ######
######## Was applied on Instagram Dataseat ##########
######## Supports VGG16 and ResNet archiectures ########
class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch):
        super(FineTuneModel, self).__init__()

        self.categories_tensor = torch.FloatTensor(1,helper.Num_category_classes).zero_()
        self.sub_categories_tensor = torch.FloatTensor(1,helper.Num_subcategories).zero_()
        self.attributes_tensor = torch.FloatTensor(1,helper.Num_attributes).zero_()
        self.categoriesDict = {}
        self.dictSubCategories_Indices = {}
        self.FillDictionaries()

        if arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.linear1 = nn.Linear(512, 512)
            self.generalCategoriesLinear = nn.Linear(512,helper.Num_category_classes)

            #  Sub categories Classification Layers
            self.SubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_subcategories)

            # Attributes Classification Layers
            self.generalAttributesLinear = nn.Linear(512,helper.Num_attributes)

            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.linear1 = nn.Linear(25088, 4096)
            self.linear2 = nn.Linear(4096, 4096)
            self.generalCategoriesLinear = nn.Linear(4096,helper.Num_category_classes)

            #  Sub categories Classification Layers
            self.SubCategoriesLinear = nn.Linear(helper.Num_category_classes,helper.Num_subcategories)

            # Attributes Classification Layers
            self.generalAttributesLinear = nn.Linear(4096,helper.Num_attributes)

            self.modelName = 'vgg16'
        else :
            raise("Finetuning not supported on this architecture yet")

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = True

    def forward(self, x,supportingCategory, supportingSubCategory, supportingAttributes):
        f = self.features(x)
        f = f.view(f.size(0), -1)

        if self.modelName == 'resnet':
            hrelu_linear = function.relu(self.linear1(f).clamp(min=0))
            number = 512
        elif self.modelName == 'vgg16':
            hrelu_linear1 = function.relu(self.linear1(f).clamp(min=0))
            hrelu_linear = function.relu(self.linear2(hrelu_linear1))
            number = 4096


        lstPossibleCategories = []
        # add the first one from the supporting category
        lstPossibleCategories.append(int(supportingCategory))

        supportingCategory = (int)(supportingCategory)
        supportingSubCategory = (int)(supportingSubCategory)

        # add the second one from the supporting sub category
        # Sub-category supporting item
        inferredCategory_fromSubCategory = self.GetCategoryFromSubcategory(supportingSubCategory)

        if inferredCategory_fromSubCategory not in lstPossibleCategories:
            lstPossibleCategories.append(int(inferredCategory_fromSubCategory))

        lstOutputs = []
        lstIndices = []

        # initializations for the correct scope of categories based
        # on chosen category
        hrelu_generalcategories = self.AdjustCategoriesLayer(lstPossibleCategories, number, hrelu_linear)
        lstIndices = self.GetSubCategoriesIndices(lstIndices, supportingCategory,supportingSubCategory)

        self.AdjustSubCategories(lstIndices,hrelu_generalcategories)

        self.AdjustAttributes(supportingAttributes, number, hrelu_linear)# append the attributes tensor to the list of outputs

        lstOutputs.append(self.categories_tensor)
        lstOutputs.append(self.sub_categories_tensor)
        lstOutputs.append(self.attributes_tensor)

        return lstOutputs

    # Adjustment of weight of connections for specific ranges of attributes
    def AdjustAttributes(self, supportingAttributes, number, hrelu_linear):
        # attributes
        dictAttributesIndices = {}
        # initializations for the correct scope of attributes based
        # on chosen category
        arr = np.empty(number)
        for i in range(0, number):
            arr[i] = helper.initialisation_attributes

        # Blouses
        lstSupportAttributes = []
        for item in supportingAttributes:
            lstSupportAttributes.append(int(item))
        indx = 1
        with open(helper.attributesClassificationsPath, "r") as file_attributes:
            for line in file_attributes:
                dictAttributesIndices[indx] = (int(line.split(",")[1]))
                indx += 1
            for counter in range(1, helper.Num_attributes):
                if dictAttributesIndices[counter] in lstSupportAttributes:  # when a supporting attribute is sent
                    # weights of connections can be slightly increased
                    # to see the effect of supporting text values
                    self.generalAttributesLinear.weight.data[counter].requires_grad = True
                    arr_temp = np.empty(number)
                    for i in range(0, number):
                        arr_temp[i] = self.generalAttributesLinear.weight.data[counter][i]
                        arr_temp[i] += helper.weight_attributes
                    self.generalAttributesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
                else:
                    self.generalAttributesLinear.weight.data[counter].requires_grad = True
                    self.generalAttributesLinear.weight.data[counter] = torch.from_numpy(arr)
        hrelu_attributes = function.relu((self.generalAttributesLinear(hrelu_linear)))
        index = 0
        for item in range(0, helper.Num_attributes):
            self.attributes_tensor[0][index] = hrelu_attributes[0].data[index]
            index += 1

    # Adjustment of weight of connections for specific ranges of sub-categories
    def AdjustSubCategories(self, lstIndices,hrelu_generalcategories):
        # initializations for the correct scope of attributes based
        # on chosen category
        arr = np.empty(helper.Num_category_classes)
        for i in range(0, helper.Num_category_classes):
            arr[i] = helper.initialisation_categories
        for counter in range(0, helper.Num_subcategories):
            # weights of unrelated attributes can be zero
            # required gradient is set to false
            if counter not in lstIndices:
                self.SubCategoriesLinear.weight.data[counter] = torch.zeros(helper.Num_category_classes)
                self.SubCategoriesLinear.weight.data[counter].requires_grad = False
            else:
                if counter in lstIndices:  # when a supporting attribute is sent
                    # weights of connections can be slightly increased
                    # to see the effect of supporting text values
                    self.SubCategoriesLinear.weight.data[counter].requires_grad = True
                    arr_temp = np.empty(helper.Num_category_classes)
                    for i in range(0, helper.Num_category_classes):
                        arr_temp[i] = self.SubCategoriesLinear.weight.data[counter][i]
                        arr_temp[i] += helper.weight_subcategories
                    self.SubCategoriesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
                else:
                    self.SubCategoriesLinear.weight.data[counter].requires_grad = True
                    self.SubCategoriesLinear.weight.data[counter] = torch.from_numpy(arr)
        hrelu_subcategories = function.relu((self.SubCategoriesLinear(hrelu_generalcategories)))
        index = 0
        for item in range(0,helper.Num_subcategories):
            self.sub_categories_tensor[0][index] = hrelu_subcategories[0].data[index]
            index += 1

    # Adjustment of weight of connections for specific ranges of categories
    def AdjustCategoriesLayer(self, lstPossibleCategories, number, hrelu_linear):
        arr = np.empty(number)
        for i in range(0, number):
            arr[i] = helper.initialisation_categories
        for counter in range(0, helper.Num_category_classes):
            # weights of unrelated categories can be zero
            # required gradient is set to false
            if counter not in lstPossibleCategories:
                self.generalCategoriesLinear.weight.data[counter] = torch.zeros(number)
                self.generalCategoriesLinear.weight.data[counter].requires_grad = False
            else:
                if counter in lstPossibleCategories:  # when a supporting category is sent or inferred
                    # weights of connections can be slightly increased
                    # to see the effect of supporting text values
                    self.generalCategoriesLinear.weight.data[counter].requires_grad = True
                    arr_temp = np.empty(number)
                    for i in range(0, number):
                        arr_temp[i] = self.generalCategoriesLinear.weight.data[counter][i]
                        arr_temp[i] += helper.weight_categories
                    self.generalCategoriesLinear.weight.data[counter] = torch.from_numpy(arr_temp)
                else:
                    self.generalCategoriesLinear.weight.data[counter].requires_grad = True
                    self.generalCategoriesLinear.weight.data[counter] = torch.from_numpy(arr)
        hrelu_generalcategories = function.relu((self.generalCategoriesLinear(hrelu_linear)))
        index = 0
        for item in range(0,helper.Num_category_classes):
            self.categories_tensor[0][index] = hrelu_generalcategories[0].data[index]
            index += 1
        return hrelu_generalcategories

    # Retrieve indices of the sub-categories based on supporting guides from category and sub-category
    def GetSubCategoriesIndices(self, lstIndices, supportingCategory, supportingSubCategory):

        if self.categoriesDict[supportingCategory] == helper.Blouses_and_tunics or self.dictSubCategories_Indices[supportingSubCategory] == helper.Blouses_and_tunics:
            lstIndices = self.GetIndices(lstIndices,helper.blouses_subcategoriesFilePath)

        if self.categoriesDict[supportingCategory] == helper.Coats or self.dictSubCategories_Indices[supportingSubCategory] == helper.Coats:
            lstIndices = self.GetIndices(lstIndices,helper.coats_subcategoriesFilePath)

        if self.categoriesDict[supportingCategory] == helper.Dresses or self.dictSubCategories_Indices[supportingSubCategory] == helper.Dresses:
            lstIndices = self.GetIndices(lstIndices,helper.dresses_subcategoriesFilePath)

        if self.categoriesDict[supportingCategory] == helper.Jeans or self.dictSubCategories_Indices[supportingSubCategory] == helper.Jeans:
            lstIndices = self.GetIndices(lstIndices,helper.jeans_subcategoriesFilePath)

        if self.categoriesDict[supportingCategory] == helper.Jackets or self.dictSubCategories_Indices[supportingSubCategory] == helper.Jackets:
            lstIndices = self.GetIndices(lstIndices,helper.jackets_subcategoriesFilePath)

        if self.categoriesDict[supportingCategory] == helper.Jumpers_and_cardigans or self.dictSubCategories_Indices[supportingSubCategory] == helper.Jumpers_and_cardigans:
            lstIndices = self.GetIndices(lstIndices,helper.jumpers_subcategoriesFilePath)

        if self.categoriesDict[supportingCategory] == helper.Skirts or self.dictSubCategories_Indices[supportingSubCategory] == helper.Skirts:
            lstIndices = self.GetIndices(lstIndices,helper.skirts_subcategoriesFilePath)

        if self.categoriesDict[supportingCategory] == helper.Tichts_and_socks or self.dictSubCategories_Indices[supportingSubCategory] == helper.Tichts_and_socks:
            lstIndices = self.GetIndices(lstIndices,helper.tights_subcategoriesFilePath)

        if self.categoriesDict[supportingCategory] == helper.Tops_and_tshirts or self.dictSubCategories_Indices[supportingSubCategory] == helper.Tops_and_tshirts:
            lstIndices = self.GetIndices(lstIndices,helper.tops_subcategoriesFilePath)

        if self.categoriesDict[supportingCategory] == helper.Trouser_and_shorts or self.dictSubCategories_Indices[supportingSubCategory] == helper.Trouser_and_shorts:
            lstIndices = self.GetIndices(lstIndices,helper.trousers_subcategoriesFilePath)

        if self.categoriesDict[supportingCategory] == helper.Shoes or self.dictSubCategories_Indices[supportingSubCategory] == helper.Shoes:
            lstIndices = self.GetIndices(lstIndices,helper.shoes_subcategoriesFilePath)

        if self.categoriesDict[supportingCategory] == helper.Bags or self.dictSubCategories_Indices[supportingSubCategory] == helper.Bags:
            lstIndices = self.GetIndices(lstIndices,helper.bags_subcategoriesFilePath)

        if self.categoriesDict[supportingCategory] == helper.All_accessories or self.dictSubCategories_Indices[supportingSubCategory] == helper.All_accessories:
            lstIndices = self.GetIndices(lstIndices,helper.accessories_subcategoriesFilePath)

        lstIndices = sorted(lstIndices, key=int)
        return lstIndices

    # Retrieves indices from subcategory file
    def GetIndices(self, lstIndices, subcategoryFile):
        indx = 0
        with open(helper.subcategoriesClassifiedPath + subcategoryFile,"r") as file_subcategories:
            for line in file_subcategories:
                lstIndices.append(int(line))  # index of the subcategory
                indx += 1
        return lstIndices

    # In this method, we have a subcategory id and we want to get the parent category id that it belongs to
    def GetCategoryFromSubcategory(self,supportingSubCategory):

        # Reading from the file to get the parent category id then break
        categoryID = -1
        with open(helper.categoriesSubcategoriesMappings, "r") as file_mappings:
            for line in file_mappings:
                subcategory_id = int(line.split(",")[1])
                parentcategory_id = int(line.split(",")[2])
                if supportingSubCategory == subcategory_id:
                    categoryID = parentcategory_id
                    break
        return categoryID

    # This method fills the categories dictionaries in the following way:
    # dictionary["blouses_and_tunics"] = 1  (key = category text, value = category id)
    # And it fills the subcategories mapping to category dictionary as follows:
    # dictionary[1] = "blouses_and_tunics" (key = subcatgory id, value = category text)
    def FillDictionaries(self):
        with open(helper.categoriesIndiciesFilePath,"r") as file_categories:
            for line in file_categories:
                self.categoriesDict[int(line.split(",")[1])] = line.split(",")[0]
        with open(helper.categoriesSubcategoriesMappings, "r") as file_subcategories:
            for line in file_subcategories:
                self.dictSubCategories_Indices[int(line.split(",")[1])] = line.split(",")[3]

# KTH - Royal Institute of Technology
# 2018

from __future__ import print_function, division
import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
import sys
import Models.Constants_DeepFashion.Helper as helper

# Helper for reading meta-data, partitioning images and other methods
# For DeepFashion Dataset
# it has categories and attributes without subcategories
class DeepFashionDataHelper(Dataset):

    # Data structure
    # {'image': image, 'clothing-categories': categories, 'clothing-attributes': attributes}
    def readGeneralCategoriesList(self):
        with open(os.path.join(self.annotations_dir,helper.category_file_path), "r") as file_category_cloth:
            self.categoryDict = {}
            counter = 0
            categoryID = 1
            for line in file_category_cloth:
                if counter < 2:
                    counter += 1
                    continue
                self.categoryDict[categoryID] =line.split(",")[0]
                categoryID += 1

    # Read attributes
    def readGeneralAttributesList(self):
        with open(os.path.join(self.annotations_dir,helper.attribute_category_file_path), "r") as file_attributes_cloth:
            self.attributesDict = {}
            counter = 0
            attributeID = 1
            for line in file_attributes_cloth:
                if counter < 2:
                    counter += 1
                    continue
                self.attributesDict[attributeID] =line.split(",")[0]
                attributeID += 1

    # Read attributes - Redundant
    def readAttributes(self):
        with open(os.path.join(self.annotations_dir,helper.attribute_category_file_path), "r") as file_attr_cloth:
            attributes_indices = {}
            counter = 0
            attribute_index = 0
            for line in file_attr_cloth:
                # skip first two lines
                if counter == 0 or counter == 1:
                    counter += 1
                    continue

                attributes_indices[line.split(",")[0].strip().lower()] = attribute_index
                attribute_index += 1

        return attributes_indices

    # read the partitioned images train,val
    def partitionImages(self, evaluation_dir):
        with open(os.path.join(evaluation_dir,helper.eval_partition_file_path), "r") as file_eval_partition:
            counter = 0
            self.trainImages = []
            self.valImages = []
            self.testImages = []
            for line in file_eval_partition:
                if counter < 2:
                    counter += 1
                    continue
                imagePath =line.split(",")[0]
                imageType = line.split(",")[1]

                if(imageType.strip() == 'train'):
                    self.trainImages.append(imagePath)
                elif(imageType.strip() == 'val'):
                    self.valImages.append(imagePath)
                elif(imageType.strip() == 'test'):
                    self.testImages.append(imagePath)

    # helper
    def key_for_value(self,dict, key):
        """Return a key in `d` having a value of `value`."""
        for k, v in dict.iteritems():
            if k == key:
                return v

    # read image category by image path
    def readImageCategory(self, imagePath):
        with open(os.path.join(self.annotations_dir,helper.category_images_file_path), "r") as file_image_category:
            counter = 0
            categoryID = -1
            for line in file_image_category:
                if counter < 2:
                    counter += 1
                    continue
                imageLoc = line.split(",")[0]
                if(imagePath == imageLoc):
                    categoryID = line.split(",")[1].strip()
        list = []
        categoryTensor = torch.FloatTensor(helper.Num_category_classes).zero_()
        for i in range (0,helper.Num_category_classes):
            if int(categoryID) == i:
                categoryTensor[i] = 1.0
                list.append(i)
            else:
                categoryTensor[i] = 0.0

        if categoryID == -1:
            print(imagePath)

        return categoryTensor

    # read image attributes by image path
    def readImageAttributes(self, imagePath):
        attributes_indices = self.readAttributes()
        attributes_line = imagePath.split("/")[1]
        separate_attributes = attributes_line.split("_")
        lstIndices = []
        for attr in separate_attributes:
            val = attr.strip().lower()
            if(attributes_indices.__contains__(val)):
                lstIndices.append(attributes_indices[val])
        attributesTensor = torch.FloatTensor(helper.Num_attribute_classes_general).zero_()
        for i in range (0,helper.Num_attribute_classes_general):
            if any(x == i for x in lstIndices):
                    attributesTensor[i] = 1.0
            else:
                    attributesTensor[i] = 0.0
        return attributesTensor

    # Intialisations - filling dictionairies, partioning images ..
    def __init__(self, root_dir_images, annotations_dir, evaluation_dir, transform=None, mode='train'):
        # read the input files data here ...
        """
        Args:
            annotations_dir (string): Path to the files with annotations.
            root_dir_images (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Read categories and attributes into dictionaries
        self.readGeneralCategoriesList(annotations_dir)
        self.readGeneralAttributesList(annotations_dir)
        self.partitionImages(evaluation_dir,root_dir_images)
        self.mode = mode

        self.annotations_dir = annotations_dir
        self.root_dir_images = root_dir_images
        self.transform = transform

    def __len__(self):
        if(self.mode == 'train'):
            length = len(self.trainImages)
        elif (self.mode == 'val'):
            length= len(self.valImages)
        return length

    def __getitem__(self, index):

        # Reading images here is more memory efficient
        # Images are not stored in memory at once, but read as required
        counter = 0


        listImages = []
        if(self.mode == 'train'):
            listImages = self.trainImages
        elif (self.mode == 'val'):
            listImages= self.valImages

        for imagePath in listImages:
            if index != counter:
                counter += 1
                continue
            img_name = os.path.join(self.root_dir_images,imagePath)
            try:
                image = Image.open(img_name)
            except:
                print("Unexpected error:", sys.exc_info()[0])
                print(img_name)
                continue


            preprocess = transforms.Compose([
                        transforms.ToTensor()
            ])
            image = np.array(image)

            image = transform.resize(image, (224, 224, 3), preserve_range=True, mode='constant')

            image -= np.array([123.68, 116.78, 103.94])

            img_tensor = preprocess(image)

            attributes = self.readImageAttributes(imagePath)
            category = self.readImageCategory(imagePath)

            break

        return img_tensor, category, attributes

    def getItemByPath(self, imagePath):
        # Reading images here is more memory efficient
        # Images are not stored in memory at once, but read as required

        img_name = os.path.join(self.root_dir_images, imagePath)
        image = io.imread(img_name)
        attributes = self.readImageAttributes(imagePath)
        category = self.readImageCategory(imagePath)
        sample = {'image': image, 'attributes': attributes, 'category': category}

        if self.transform:
            sample = self.transform(sample)

        sample = {'image': sample['image'], 'attributes': attributes, 'category': category}

        return sample

    def returnTrainImageslist(self):
        return self.trainImages

    def iterateOverTrainImages(self):
        for imagePath in self.trainImages:
            img_name = os.path.join(self.root_dir_images, imagePath)
            image = io.imread(img_name)
            attributes = self.readImageAttributes(imagePath)
            category = self.readImageCategory(imagePath)
            trainImage = {'image': image, 'attributes': attributes, 'category': category}

            if self.transform:
                trainImage = self.transform(trainImage)

    def iterateOverValImages(self):
        for imagePath in self.valImages:
            img_name = os.path.join(self.root_dir_images, imagePath)
            image = io.imread(img_name)
            attributes = self.readImageAttributes(imagePath)
            category = self.readImageCategory(imagePath)
            valImages = {'image': image, 'attributes': attributes, 'category': category}

            if self.transform:
                trainImage = self.transform(valImages)

    def iterateOverTestImages(self):
        for imagePath in self.valImages:
            img_name = os.path.join(self.root_dir_images, imagePath)
            image = io.imread(img_name)
            attributes = self.readImageAttributes(imagePath)
            category = self.readImageCategory(imagePath)
            testImage = {'image': image, 'attributes': attributes, 'category': category}

            if self.transform:
                testImage = self.transform(testImage)

# Not Used
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        new_h = self.output_size
        new_w = self.output_size
        img = transforms.Resize(new_h, new_w)

        return  img

# Not Used
class Resize(object):
    def __init__(self, new_width, new_height, _R_Mean, _G_Mean, _B_Mean):
        self.new_width = new_width
        self.new_height = new_height
        self._R_Mean = _R_Mean
        self._G_Mean = _G_Mean
        self._B_Mean = _B_Mean


    def __call__(self, sample, attributes, category):

        image = transform.resize(sample['image'], (self.new_width, self.new_height, 3), preserve_range=True, mode='constant')

        image -= np.array([self._R_Mean, self._G_Mean, self._B_Mean])

        image = np.moveaxis(image, 2, 0)

        image = image[None]
        image = torch.from_numpy(image).float()  # convert the numpy array into torch tensor
        image = Variable(image)
        return {'image': image, 'attributes': attributes, 'category': category}



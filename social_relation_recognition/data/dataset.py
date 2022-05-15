"""
	Custom Pytorch dataset.
"""


import os

import torch
import torch.utils.data as data
from PIL import Image
from tools.utils_dataset import (fix_bounding_box, get_bounding_boxes,
                                 get_context_bounding_box)


class SRDataset(data.Dataset):

    def __init__(self, path_images, ids_images, metadata, type, transform_personal, transform_local, transform_global):

        super(SRDataset, self).__init__()

        self.path_images = path_images
        self.ids_images = ids_images
        self.metadata = metadata

        self.type = type
        self.transform_personal = transform_personal
        self.transform_local = transform_local
        self.transform_global = transform_global

    def __getitem__(self, index):
        """Return personal, local and global data per image."""

        # Images lists
        list_ids_images = []
        list_personal_images = []
        list_local_images = []

        # Tensor outputs
        list_personal_tensors = []
        list_local_tensors = []
        list_labels_tensors = []

        # Get image ID and image metadata
        id_image = self.ids_images[index]
        str_id_image = id_image.zfill(5) + '.jpg'
        data_image = self.metadata[id_image]

        # Get image
        path_image = os.path.join(self.path_images, str_id_image)
        image_global = Image.open(path_image).convert('RGB')

        # Get personal and local scale images
        for relation, label in data_image[self.type].items():

            # Get personal keys
            split_space = relation.split()
            id_personal_1 = split_space[0]
            id_personal_2 = split_space[1]

            # Get relation persons and context boxes
            bounding_box_person_1, bounding_box_person_2 = get_bounding_boxes(
                relation, data_image['body_bbox'])
            context_bounding_box = get_context_bounding_box(
                bounding_box_person_1, bounding_box_person_2)

            if not(id_personal_1 in list_ids_images):
                list_ids_images.append(id_personal_1)
                fixed_bounding_box = fix_bounding_box(
                    bounding_box_person_1, image_global.size)
                personal_image = image_global.crop((fixed_bounding_box))
                list_personal_images.append(personal_image)

            if not(id_personal_2 in list_ids_images):
                list_ids_images.append(id_personal_2)
                fixed_bounding_box = fix_bounding_box(
                    bounding_box_person_2, image_global.size)
                personal_image = image_global.crop((fixed_bounding_box))
                list_personal_images.append(personal_image)

            # Fix box
            fixed_context_bounding_box = fix_bounding_box(
                context_bounding_box, image_global.size)

            # Get image
            image_local = image_global.crop((fixed_context_bounding_box))
            list_local_images.append(image_local)

            # Get label
            list_labels_tensors.append(torch.tensor(label))

        # Apply transformations
        for image in list_personal_images:
            tensor_image = self.transform_personal(image)
            list_personal_tensors.append(tensor_image)

        # Apply transformations
        for image in list_local_images:
            tensor_image = self.transform_personal(image)
            list_local_tensors.append(tensor_image)

        tensor_global = self.transform_global(image_global)

        # Get tensors
        tensor_personal = torch.stack(list_personal_tensors)
        tensor_local = torch.stack(list_local_tensors)
        tensor_label = torch.stack(list_labels_tensors)

        return id_image, tensor_personal, tensor_local, tensor_global, tensor_label

    def __len__(self):
        return len(self.ids_images)

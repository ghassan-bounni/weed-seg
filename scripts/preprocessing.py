import torch


def create_binary_masks(mask, num_classes):
    binary_masks = torch.zeros(
        (num_classes, mask.shape[0], mask.shape[1]),
        dtype=torch.uint8,
    )
    for class_id in range(num_classes):
        binary_masks[class_id] = (mask == class_id).type(torch.uint8)

    return binary_masks

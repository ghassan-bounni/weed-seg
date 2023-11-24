import os

import cv2
import torch
import torch.nn.functional as F
import torchmetrics.functional.classification as metrics
from torchinfo import summary

from models.base import BaseModel
from data.dataset import create_dataloader

import logging
from utils.utils import (
    load_checkpoint,
    get_stem_coordinates,
    save_data_to_json,
)


def test(
    model_config: dict,
    test_config: dict,
    data_path: str,
    checkpoint_path: str,
    output_path: str,
):
    """
    Evaluates the model.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = logging.getLogger("StemDetectionLogger")
    masks_dir = os.path.join(output_path, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    (
        batch_size,
        num_workers,
        eval_transforms,
    ) = test_config.values()

    eval_dataloader = create_dataloader(
        data_path=os.path.join(data_path, "test"),
        transforms_dict=eval_transforms,
        batch_size=batch_size,
        shuffle=False,
        num_classes=model_config["out_channels"],
        num_workers=num_workers,
    )

    model = BaseModel(**model_config)
    model.to(device)

    load_checkpoint("eval", model, ckpt_path=checkpoint_path)
    summary(model, input_size=(batch_size, *eval_dataloader.dataset[0][0].shape))

    model.eval()
    with torch.inference_mode():
        accuracy = 0
        stem_data = {"results": {}}
        for batch_idx, (inputs, targets, img_ids) in enumerate(eval_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # forward pass
            logits = model(inputs)
            output = F.softmax(logits, dim=1).exp()

            # getting the class with the highest probability per pixel in each image
            output_argmax = output.argmax(1)

            # calculating the accuracy
            accuracy += metrics.multiclass_accuracy(
                output_argmax, targets, num_classes=output.shape[1]
            ).item()

            # saving the masks
            for img_id, img in zip(img_ids, output_argmax):
                mask_path = os.path.join(masks_dir, f"{img_id.split('.')[0]}.png")
                cv2.imwrite(mask_path, img.cpu().numpy())

            # contains the coordinates of the stems in each image
            stem_coordinates = get_stem_coordinates(
                output_argmax, output.shape[1], img_ids
            )
            stem_data["results"].update(stem_coordinates)

        save_data_to_json(stem_data, "stem_data.json", output_path)

        accuracy /= len(eval_dataloader)
        logger.info(f"Accuracy: {(accuracy * 100):.2f}%")

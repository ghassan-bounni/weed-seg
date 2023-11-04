import torch
from torchinfo import summary

from models.base import BaseModel
from data.dataset import create_dataloader

import logging
from utils.utils import (
    load_criterion,
    load_checkpoint,
    get_stem_coordinates,
    save_data_to_json,
)


def test(
    model_config: dict,
    test_config: dict,
    checkpoint_name: str,
    seed: int,
    output_dir: str,
):
    """
    Evaluates the model.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = logging.getLogger("StemDetectionLogger")

    (
        batch_size,
        loss_fn,
        eval_transforms,
    ) = test_config.values()

    eval_dataloader = create_dataloader(
        "data/test/", eval_transforms, batch_size, False, seed
    )

    model = BaseModel(**model_config)
    model.to(device)

    criterion = load_criterion(loss_fn)

    load_checkpoint("eval", model, checkpoint_name=checkpoint_name)
    summary(model, input_size=(batch_size, *eval_dataloader.dataset[0][0].shape))

    model.eval()
    with torch.inference_mode():
        accuracy = 0
        stem_data = {"results": []}
        for batch_idx, (inputs, targets, img_ids) in enumerate(eval_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            output = model(inputs)

            loss = criterion(output, targets)
            logger.info(f"loss: {loss}")

            # getting the class with the highest probability per pixel in each image
            output_argmax = output.argmax(1)

            accuracy += (output_argmax == targets).float().mean()

            # contains the coordinates of the stems in each image
            stem_coordinates = get_stem_coordinates(
                output_argmax, output.shape[1], img_ids
            )
            stem_data["results"].extend(stem_coordinates)

        save_data_to_json(stem_data, "stem_data.json", output_dir)

        accuracy /= len(eval_dataloader)
        logger.info(f"Accuracy: {accuracy}")

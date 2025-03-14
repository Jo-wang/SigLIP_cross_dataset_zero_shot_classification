import random

import torch
import torch.nn.functional as F
import argparse
import numpy as np
from tqdm import tqdm
import open_clip
from utils import build_test_data_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="fgvc",
        help="Dataset to process",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/user/data",
        help="Path to the datasets directory.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ViT-B-16-SigLIP",
        help="Model checkpoint to use.",
    )


    args = parser.parse_args()
    return args



@torch.no_grad
def run_test(data_loader, model, text_token):
    accuracies = []

    text_features = model.encode_text(text_token)
    text_features = F.normalize(text_features, dim=-1)

    for i, (images, target) in enumerate(
        tqdm(data_loader, desc="Processed test images: ")
    ):
        image_features = model.encode_image(
            images.to(device)
        )
        image_features = F.normalize(image_features, dim=-1)
        model_logits = (
            image_features @ text_features.T * model.logit_scale.exp()
            + model.logit_bias
        )
        
        probs = torch.sigmoid(model_logits)
        text_pred = probs.argmax(dim=-1).item()
        acc = int(text_pred == target.item())
        accuracies.append(acc)

        if i % 500 == 0:
            print(
                "- Accuracy: {:.2f}. -\n".format(100 * (sum(accuracies) / len(accuracies)))
            )
    final_acc = sum(accuracies) / len(accuracies)
    print("The accuracy is", final_acc)


def main():
    cfg = get_arguments()
    
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        cfg.ckpt, pretrained="webli"
    )
    tokenizer = open_clip.get_tokenizer(cfg.ckpt)
    test_loader, classnames, _ = build_test_data_loader(
        cfg.dataset, cfg.data_root, preprocess_val
    )
    candidate_labels = [f"a photo of a {label}" for label in classnames]
    text_token = tokenizer(candidate_labels, context_length=model.context_length).to(
        device
    )
    model.to(device)
    run_test(test_loader, model, text_token)



if __name__ == "__main__":
    main()


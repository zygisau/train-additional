import torch
import yaml

from prithvi.Prithvi import MaskedAutoencoderViT


class AppendFeatures:
    def __init__(self, feature_model_path, feature_model_checkpoint_path):
        weights_path = feature_model_path
        # load weights
        if torch.cuda.is_available():
            checkpoint = torch.load(weights_path, map_location="cuda:0")
            device = torch.device("cuda")
            checkpoint.to(device)
        else:
            checkpoint = torch.load(weights_path, map_location="cpu")

        # read model config
        model_cfg_path = feature_model_checkpoint_path
        with open(model_cfg_path) as f:
            model_config = yaml.safe_load(f)

        model_args = model_config["model_args"]

        # let us use only 1 frame for now (the model was trained on 3 frames)
        model_args["num_frames"] = 1

        # instantiate model
        model = MaskedAutoencoderViT(**model_args)
        model.eval()

        # load weights into model
        # strict=false since we are loading with only 1 frame, but the warning is expected
        del checkpoint['pos_embed']
        del checkpoint['decoder_pos_embed']
        _ = model.load_state_dict(checkpoint, strict=False)
        self.model = model

    def __call__(self, samples):
        img1, img2, mask = samples
        # only take b,g,r,nir,swir
        img1_reshaped = img1[:, [2, 1, 0, 3, 8, 9], :, :]
        img2_reshaped = img2[:, [2, 1, 0, 3, 8, 9], :, :]
        img1_reshaped = img1_reshaped.reshape(
            *img1_reshaped.shape[:2], 1, *img1_reshaped.shape[-2:]).float()
        img2_reshaped = img2_reshaped.reshape(
            *img2_reshaped.shape[:2], 1, *img2_reshaped.shape[-2:]).float()
        mask_ratio = 0.75
        with torch.no_grad():
            features1, _, __ = self.model(img1_reshaped, mask_ratio)
            features2, _, __ = self.model(img2_reshaped, mask_ratio)

        features1 = torch.flatten(features1, 1)
        features2 = torch.flatten(features2, 1)
        return img1, img2, mask, features1, features2

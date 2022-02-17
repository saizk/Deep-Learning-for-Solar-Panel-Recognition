import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp


class SolarPanelsModel(pl.LightningModule):

    def __init__(self, arch, encoder, in_channels, out_classes, model_params, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder,
            in_channels=in_channels, classes=out_classes, **kwargs
        )

        params = smp.encoders.get_preprocessing_params(encoder)  # preprocessing parameters for image
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.lr = model_params.get('lr') or 0.0001
        self.loss_fn = model_params.get('loss') or smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True),
        self.optimizer = model_params.get('optimizer') or torch.optim.Adam

    def forward(self, image):
        image = (image - self.mean) / self.std  # normalize image here
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        h, w = image.shape[2:]

        assert h % 32 == 0 and w % 32 == 0

        assert mask.ndim == 4  # Shape of the mask should be [batch_size, num_classes, height, width] for binary segmentation num_classes = 1
        assert mask.max() <= 1.0 and mask.min() >= 0  # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation

        logits_mask = self.forward(image)

        loss = self.loss_fn(logits_mask, mask)  # Predicted mask contains logits, and loss_fn param `from_logits` is set to True

        prob_mask = logits_mask.sigmoid()  # first convert mask values to probabilities
        pred_mask = (prob_mask > 0.5).float()  # apply thresholding

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        per_image_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")

        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        dataset_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_per_image_f1": per_image_f1,
            f"{stage}_dataset_f1": dataset_f1,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

def rebuild_pruned_unet(model, masks):
    """
    Use masks to determine new channel counts per encoder/decoder block.
    Return a new UNet instance with updated layer widths.
    """


def extract_architecture_from_masks(masks):
    """Return encoder_features, bottleneck_features, decoder_features lists."""

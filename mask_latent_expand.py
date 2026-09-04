import torch

BLOCK = 8


class LatentMaskToFramesLTX:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"masks": ("MASK",)}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "expand"
    CATEGORY = "Zalotron"
    DESCRIPTION = (
        "Turns one mask per LTX latent back into a per-frame mask batch, "
        "repeating each latent's mask across the frames it covers. Latent 0 "
        "yields frame 0 alone, each following latent yields 8 frames, so L "
        "latents give 1 + 8*(L-1) frames."
    )

    def expand(self, masks):
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)

        # Latent 0 stands for frame 0 alone; the rest cover 8 frames each.
        frames = [masks[0:1]]
        for i in range(1, masks.shape[0]):
            frames.append(masks[i:i + 1].repeat(BLOCK, 1, 1))

        return (torch.cat(frames, dim=0),)


NODE_CLASS_MAPPINGS = {"LatentMaskToFramesLTX": LatentMaskToFramesLTX}
NODE_DISPLAY_NAME_MAPPINGS = {"LatentMaskToFramesLTX": "to Frame mask (LTX)"}

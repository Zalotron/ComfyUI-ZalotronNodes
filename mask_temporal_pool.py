import torch

BLOCK = 8


class MaskTemporalPoolLTX:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"masks": ("MASK",)}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "pool"
    CATEGORY = "Zalotron"
    DESCRIPTION = (
        "Turns a per-frame mask batch into one mask per LTX latent, taking the "
        "union of each latent's frames. Feed the result to "
        "LTXVSetVideoLatentNoiseMasks, which expects one mask per latent."
    )

    def pool(self, masks):
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)

        # The VAE trims its input to 1 + n*8 frames, so anything past that never
        # reaches a latent and must not produce a mask either.
        total = 1 + ((masks.shape[0] - 1) // BLOCK) * BLOCK

        # Frame 0 gets a latent of its own; each following latent covers 8 frames.
        pooled = [masks[0:1]]
        for start in range(1, total, BLOCK):
            pooled.append(masks[start:start + BLOCK].amax(dim=0, keepdim=True))

        return (torch.cat(pooled, dim=0),)


NODE_CLASS_MAPPINGS = {"MaskTemporalPoolLTX": MaskTemporalPoolLTX}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskTemporalPoolLTX": "to Latent mask (LTX)"}

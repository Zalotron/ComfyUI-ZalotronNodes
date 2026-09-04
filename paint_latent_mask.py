import torch


class PaintLatentMaskLTX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "index": ("STRING", {"default": "0", "tooltip": "One index, or several separated by commas: 0,1,-1"}),
                "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "paint"
    CATEGORY = "Zalotron"
    DESCRIPTION = (
        "Fills one or more latents' noise mask with a value. Index 0 is frame 0, index 1 "
        "is frames 1-8, index 2 is frames 9-16, and so on; negative indices count "
        "from the end (-1 is the last latent). If the latent has no mask yet one "
        "is created full of zeros, so chaining these marks only the latents you "
        "touch as generatable. Place it before any AddGuide."
    )

    def paint(self, latent, index, value):
        out = latent.copy()
        samples = out["samples"]
        batch, _, count, _, _ = samples.shape

        mask = out.get("noise_mask")
        if mask is None or mask.shape[2] != count:
            # No usable mask yet: fill with the opposite of what is being painted,
            # so a single node already expresses the intent (paint 0 -> rest is 1).
            # 1x1 spatially is the cheapest form and broadcasts fine.
            mask = torch.full(
                (batch, 1, count, 1, 1),
                1.0 - value,
                dtype=samples.dtype,
                device=samples.device,
            )
        else:
            # Keep whatever spatial size the mask already has.
            mask = mask.clone()

        for token in index.split(","):
            token = token.strip()
            if not token:
                continue
            i = int(token)
            if i < 0:
                i += count
            mask[:, :, max(0, min(i, count - 1))] = value

        out["noise_mask"] = mask
        return (out,)


NODE_CLASS_MAPPINGS = {"PaintLatentMaskLTX": PaintLatentMaskLTX}
NODE_DISPLAY_NAME_MAPPINGS = {"PaintLatentMaskLTX": "Paint Latent Mask (LTX)"}

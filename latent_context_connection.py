import comfy.utils
import torch

CONTEXT_FRAMES = 25
HEAD_LATENTS = 4
TAIL_LATENTS = 1


class LatentContextConnectionLTX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("VAE",),
            },
            "optional": {
                "clip_1": ("IMAGE",),
                "clip_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "connect"
    CATEGORY = "Zalotron"
    DESCRIPTION = (
        "Writes the head and/or tail of a latent with encoded context clips. "
        "clip_1 is the beginning: its first 25 frames become the first 4 latents. "
        "clip_2 is the end: its first 25 frames encode to 4 latents and only the "
        "last one is written, so the model authors the 16 frames leading into it "
        "instead of having to land on them. The noise mask comes out white except "
        "for the written latents. At least one clip must be connected."
    )

    @staticmethod
    def _encode(vae, images, width, height):
        pixels = comfy.utils.common_upscale(
            images[:CONTEXT_FRAMES].movedim(-1, 1), width, height, "bilinear", "disabled"
        ).movedim(1, -1)
        return vae.encode(pixels[:, :, :, :3])

    def connect(self, latent, vae, clip_1=None, clip_2=None):
        if clip_1 is None and clip_2 is None:
            raise ValueError("Connect at least one of clip_1 or clip_2.")

        out = latent.copy()
        samples = out["samples"].clone()
        batch, _, count, latent_height, latent_width = samples.shape

        _, height_scale, width_scale = vae.downscale_index_formula
        width = latent_width * width_scale
        height = latent_height * height_scale

        if clip_1 is not None:
            head = self._encode(vae, clip_1, width, height)
            n = min(HEAD_LATENTS, head.shape[2], count)
            samples[:, :, :n] = head[:, :, :n]

        tail_n = 0
        if clip_2 is not None:
            tail = self._encode(vae, clip_2, width, height)
            # The clip's first latent holds a single frame and cannot be placed
            # anywhere but position 0, so only the full 8-frame ones are used.
            tail_n = min(TAIL_LATENTS, tail.shape[2] - 1, count)
            samples[:, :, count - tail_n:] = tail[:, :, tail.shape[2] - tail_n:]

        mask = torch.ones(
            (batch, 1, count, 1, 1), dtype=samples.dtype, device=samples.device
        )
        if clip_1 is not None:
            mask[:, :, :min(HEAD_LATENTS, count)] = 0.0
        if tail_n:
            mask[:, :, count - tail_n:] = 0.0

        out["samples"] = samples
        out["noise_mask"] = mask
        return (out,)


NODE_CLASS_MAPPINGS = {"LatentContextConnectionLTX": LatentContextConnectionLTX}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentContextConnectionLTX": "Latent context connection (LTX)"
}

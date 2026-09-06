import comfy.utils
import torch

BLOCK = 8
TAIL_FREE = 2


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
        "clip_1 is the beginning: the whole clip is used, trimmed at the end to "
        "8n+1 frames, and every latent it produces is written and frozen. "
        "clip_2 is the end: same 8n+1 trim, then frame 0 is duplicated at the "
        "cost of the last one so the blocks land on the clip's real start, and "
        "every latent but the "
        "single-frame first one is written at the end. The mask freezes all of "
        "them except the 2 nearest the junction, which the model regenerates so "
        "it has room to bend into clip_2's motion. At least one clip must be "
        "connected."
    )

    @staticmethod
    def _encode(vae, images, width, height):
        pixels = comfy.utils.common_upscale(
            images.movedim(-1, 1), width, height, "bilinear", "disabled"
        ).movedim(1, -1)
        return vae.encode(pixels[:, :, :, :3])

    def connect(self, latent, vae, clip_1=None, clip_2=None):
        if clip_1 is None and clip_2 is None:
            raise ValueError("Connect at least one of clip_1 or clip_2.")

        out = latent.copy()
        samples = out["samples"].clone()
        _, _, count, latent_height, latent_width = samples.shape

        _, height_scale, width_scale = vae.downscale_index_formula
        width = latent_width * width_scale
        height = latent_height * height_scale

        head_n = 0
        if clip_1 is not None:
            usable = ((clip_1.shape[0] - 1) // BLOCK) * BLOCK + 1
            head = self._encode(vae, clip_1[:usable], width, height)
            head_n = min(head.shape[2], count)
            samples[:, :, :head_n] = head[:, :, :head_n]

        tail_n = 0
        if clip_2 is not None:
            usable = ((clip_2.shape[0] - 1) // BLOCK) * BLOCK + 1
            frames = clip_2[:usable]
            # Duplicating frame 0 at the cost of the last one shifts the block
            # boundaries, so the usable latents cover frames 0..N-2 instead of
            # 1..N-1 and the clip's real first frame stops being unreachable.
            frames = torch.cat([frames[:1], frames[:-1]])
            tail = self._encode(vae, frames, width, height)
            # The clip's first latent holds a single frame and cannot be placed
            # anywhere but position 0, so only the full 8-frame ones are used.
            tail_n = min(tail.shape[2] - 1, count)
            samples[:, :, count - tail_n:] = tail[:, :, tail.shape[2] - tail_n:]

        # Same construction as KJ's LTXVAudioVideoMask: start from zeros at the
        # latent's own resolution and paint the free span with ones.
        frozen = max(tail_n - TAIL_FREE, 0)
        mask = torch.zeros_like(samples)[:, :1]
        mask[:, :, head_n:count - frozen] = 1.0

        out["samples"] = samples
        out["noise_mask"] = mask
        return (out,)


NODE_CLASS_MAPPINGS = {"LatentContextConnectionLTX": LatentContextConnectionLTX}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentContextConnectionLTX": "Latent context connection (LTX)"
}

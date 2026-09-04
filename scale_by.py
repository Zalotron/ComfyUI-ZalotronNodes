import comfy.utils


class AmazeScaleBy:
    METHODS = ["area", "bilinear", "nearest-exact", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scale_by": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 8.0, "step": 0.01}),
                "upscale_method": (cls.METHODS, {"default": "area"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "scale"
    CATEGORY = "Zalotron"
    DESCRIPTION = (
        "Scales an image and/or a mask by a factor. Both inputs are optional and "
        "each is only touched if connected, so wiring just one costs nothing for "
        "the other. A scale_by of 1.0 passes everything through untouched. "
        "'area' is the cheapest and best choice for downscaling."
    )

    def scale(self, scale_by, upscale_method, image=None, mask=None):
        if scale_by == 1.0:
            return (image, mask)

        if image is not None:
            height, width = image.shape[1], image.shape[2]
            image = comfy.utils.common_upscale(
                image.movedim(-1, 1),
                max(1, round(width * scale_by)),
                max(1, round(height * scale_by)),
                upscale_method,
                "disabled",
            ).movedim(1, -1)

        if mask is not None:
            source = mask if mask.dim() == 3 else mask.unsqueeze(0)
            height, width = source.shape[-2], source.shape[-1]
            # comfy's lanczos transposes single-channel tensors (it squeezes the
            # channel and then movedims a 2D array), and it is ~60x slower than
            # bicubic for no visible gain on a mask.
            method = "bicubic" if upscale_method == "lanczos" else upscale_method
            mask = comfy.utils.common_upscale(
                source.unsqueeze(1),
                max(1, round(width * scale_by)),
                max(1, round(height * scale_by)),
                method,
                "disabled",
            ).squeeze(1)
            # bicubic overshoots outside [0, 1], which breaks masks.
            if method == "bicubic":
                mask = mask.clamp(0.0, 1.0)

        return (image, mask)


NODE_CLASS_MAPPINGS = {"AmazeScaleBy": AmazeScaleBy}
NODE_DISPLAY_NAME_MAPPINGS = {"AmazeScaleBy": "Scale by"}

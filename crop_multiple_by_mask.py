import torch


class CropEnsuringMultipleByMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "multiple_of": ("INT", {"default": 64, "min": 1, "max": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "x", "y", "width", "height")
    FUNCTION = "crop"
    CATEGORY = "Zalotron"
    DESCRIPTION = (
        "Crops image and mask to a region whose width and height are multiples "
        "of multiple_of, never cutting into the mask and never going outside the "
        "image. Grows towards whatever side has room, so a mask against an edge "
        "expands inwards. With a mask batch it uses the union of every frame, so "
        "the crop stays constant across the clip. Returns x/y/width/height to "
        "paste the result back."
    )

    @staticmethod
    def _span(low, high, target, limit):
        """Places a window of size `target` over [low, high] inside [0, limit]."""
        start = low - (target - (high - low + 1)) // 2
        return max(0, min(start, limit - target))

    def crop(self, image, mask, multiple_of):
        _, height, width, _ = image.shape

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # The crop has to be identical for every frame or the batch stops being a
        # video, so the box comes from the union of the whole mask batch.
        marked = (mask > 0).any(dim=0)
        rows = torch.nonzero(marked.any(dim=1), as_tuple=True)[0]
        cols = torch.nonzero(marked.any(dim=0), as_tuple=True)[0]

        if len(rows) == 0:
            y_min, y_max, x_min, x_max = 0, height - 1, 0, width - 1
        else:
            y_min, y_max = int(rows[0]), int(rows[-1])
            x_min, x_max = int(cols[0]), int(cols[-1])

        def size(span, limit):
            wanted = -(-span // multiple_of) * multiple_of
            if wanted <= limit:
                return wanted
            # The mask is wider than any multiple that fits; take the largest one.
            return (limit // multiple_of) * multiple_of or limit

        crop_w = size(x_max - x_min + 1, width)
        crop_h = size(y_max - y_min + 1, height)

        x = self._span(x_min, x_max, crop_w, width)
        y = self._span(y_min, y_max, crop_h, height)

        return (
            image[:, y:y + crop_h, x:x + crop_w, :],
            mask[:, y:y + crop_h, x:x + crop_w],
            x,
            y,
            crop_w,
            crop_h,
        )


NODE_CLASS_MAPPINGS = {"CropEnsuringMultipleByMask": CropEnsuringMultipleByMask}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CropEnsuringMultipleByMask": "Crop Ensuring W/H Multiple (mask)"
}

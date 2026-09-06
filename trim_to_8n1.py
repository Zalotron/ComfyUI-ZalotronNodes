BLOCK = 8


class TrimTo8n1LTX:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "trim"
    CATEGORY = "Zalotron"
    DESCRIPTION = (
        "Drops frames off the end until the batch is 8n+1 long, the only length "
        "the LTX VAE encodes without leaving a partial block. A batch that is "
        "already 8n+1 comes out untouched."
    )

    def trim(self, images):
        keep = ((images.shape[0] - 1) // BLOCK) * BLOCK + 1
        return (images[:keep],)


NODE_CLASS_MAPPINGS = {"TrimTo8n1LTX": TrimTo8n1LTX}
NODE_DISPLAY_NAME_MAPPINGS = {"TrimTo8n1LTX": "Trim to 8n+1 (LTX)"}

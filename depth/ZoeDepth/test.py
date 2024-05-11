##### sample prediction
import torch

# Zoe_N
model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

# # Local file
from PIL import Image
# image = Image.open("../output_image.jpeg").convert("RGB")  # load
# depth_numpy = zoe.infer_pil(image)  # as numpy

# depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image

# depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor

# # Tensor 
# from zoedepth.utils.misc import pil_to_batched_tensor
# X = pil_to_batched_tensor(image).to(DEVICE)
# depth_tensor = zoe.infer(X)

# # From URL
# from zoedepth.utils.misc import get_image_from_url

# # Example URL
# URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS4W8H_Nxk_rs3Vje_zj6mglPOH7bnPhQitBH8WkqjlqQVotdtDEG37BsnGofME3_u6lDk&usqp=CAU"


image = Image.open("../output_image.png").convert("RGB")  # load
depth = zoe.infer_pil(image)  # as numpy

# Save raw
from zoedepth.utils.misc import save_raw_16bit
fpath = "../output.png"
save_raw_16bit(depth, fpath)

# Colorize output
from zoedepth.utils.misc import colorize

colored = colorize(depth)

# save colored output
fpath_colored = "../output_colored.png"
Image.fromarray(colored).save(fpath_colored)
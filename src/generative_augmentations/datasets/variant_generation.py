from typing import cast
from pathlib import Path
from PIL import Image
import re
import logging
from itertools import batched
import shutil

import tyro
import torch as th
from torchvision.transforms.functional import to_tensor, resize
from torchvision.utils import save_image
from tqdm import tqdm, trange
from pycocotools.mask import decode
from transformers.models.paligemma import PaliGemmaForConditionalGeneration
from transformers.pipelines import pipeline
from transformers.pipelines.depth_estimation import DepthEstimationPipeline
from transformers.pipelines.text_generation import TextGenerationPipeline
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.image_processing_auto import AutoImageProcessor
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from controlnet_aux.processor import HEDdetector
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from src.generative_augmentations.datasets.coco import index_to_name
from src.generative_augmentations.config import Config


# TODO:
# handle upscaling later
# Upscale image if necessary for small objects/bboxes
#   Remember to upscale bboxes and segmentations


# Estimate depth for image
# estimate edge map for image

# for each bounding box
#     generate prompt
#     generate n variants, controlnet depth edge
#     cut out bounding box
#     downscale to original size
#     Store in a way that is tied to bounding box


logger = logging.getLogger(__name__)


class VariantGeneration:
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path | None = None,
        num_variants: int = 3,
        bbox_min_side_length: int = 75,
        device: th.device | None = None
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir if output_dir else input_dir
        self.num_variants = num_variants
        self.bbox_min_side_length = bbox_min_side_length

        self.device = device if device else th.device("cuda" if th.cuda.is_available() else "cpu")
        self.dtype = th.bfloat16 if th.cuda.is_available() else th.float32

        logger.info(f"Loading models...")

        # Based upon: https://arxiv.org/abs/2406.09414
        self.depth_model = self._get_depth_model()
        # Based upon: https://arxiv.org/abs/2302.05543
        self.edge_model = HEDdetector.from_pretrained("lllyasviel/Annotators").to(self.device)
        # Based upon: https://arxiv.org/abs/2407.07726
        self.vqa_model, self.vqa_model_tokenizer = self._get_vqa_model()
        # Based upon: https://arxiv.org/abs/2407.10671
        self.prompt_model = self._get_prompt_model()
        # Based upon: https://arxiv.org/abs/2307.01952
        self.diffusion_model = self._get_diffusion_model()
        
        logger.info(f"Loaded models")


        # Load names of image directories
        with open(self.input_dir / "image_names.txt", 'r') as file:
            self.img_dirs = file.readline().strip().split(" ")


    def _get_depth_model(self) -> DepthEstimationPipeline:
        # Remove info logging
        pipe_logger = logging.getLogger("transformers.pipelines.base")
        pipe_logger_level_orig = pipe_logger.level
        pipe_logger.setLevel(logging.ERROR)

        # Get depth model pipeline
        pipe = cast(
            DepthEstimationPipeline,
            pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Large-hf",
                device_map=self.device,
                image_processor=AutoImageProcessor.from_pretrained(
                    "depth-anything/Depth-Anything-V2-Large-hf",
                    _from_pipeline="depth-estimation", 
                    device_map=self.device,
                    torch_dtype=self.dtype,
                    use_fast=False,
                )
            )
        )

        # Reset logger level
        pipe_logger.setLevel(pipe_logger_level_orig)

        # Return pipeline
        return pipe


    def _get_vqa_model(self) -> tuple[PaliGemmaForConditionalGeneration, AutoProcessor]: 
        # Get the model 
        model_id = "google/paligemma2-3b-mix-224"
        vqa_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=self.dtype, 
            device_map=self.device
        ).eval()

        vqa_tokenizer = AutoProcessor.from_pretrained(model_id)

        return vqa_model, vqa_tokenizer


    def _get_prompt_model(self) -> TextGenerationPipeline:
        # Remove info logging
        pipe_logger = logging.getLogger("transformers.pipelines.base")
        pipe_logger_level_orig = pipe_logger.level
        pipe_logger.setLevel(logging.ERROR)

        # Get depth model pipeline
        pipe = cast(
            TextGenerationPipeline,
            pipeline(task="text-generation", model="Qwen/Qwen2-1.5B-Instruct", torch_dtype=self.dtype, device_map=self.device)
        )

        # Reset logger level
        pipe_logger.setLevel(pipe_logger_level_orig)

        # Return pipeline
        return pipe


    def _get_diffusion_model(self) -> StableDiffusionXLControlNetInpaintPipeline:
        controlnet_depth = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            use_safetensors=True,
            torch_dtype=self.dtype,
        )
        controlnet_edge = ControlNetModel.from_pretrained(
            "SargeZT/controlnet-sd-xl-1.0-softedge-dexined",
            # "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=self.dtype,
        )
        
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=self.dtype)
        
        diffusion_model = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=[controlnet_depth, controlnet_edge],
            vae=vae,
            torch_dtype=self.dtype
        )

        diffusion_model.set_progress_bar_config(disable=True)
        diffusion_model.enable_model_cpu_offload()

        # return diffusion_model.to(self.device)
        return diffusion_model


    def _estimate_depth(self, img: Image.Image) -> th.Tensor:
        return self.depth_model(img)["predicted_depth"].to(self.device) # type: ignore


    def _estimate_edges(self, img: Image.Image) -> th.Tensor:
        return th.tensor(self.edge_model(img, output_type="numpy")).permute(2, 0, 1).to(device=self.device, dtype=th.float32) / 255.0


    def _describe_object(self, img: th.Tensor) -> str:
        # NOTE: If the bounding box contains multiple objects,
        #  this method of obtaining a description will cause issues.
        #  In such a case, maybe cut out the instance via the mask first.
        
        # Prepare prompt
        prompt = "<image>describe en"
        inputs = self.vqa_model_tokenizer( # pyright: ignore[ reportCallIssue ]
            text=prompt,
            images=img * 255.0,
            return_tensors="pt"
        ).to(self.vqa_model.device, self.vqa_model.dtype)
        inputs_len = inputs['input_ids'].shape[-1]

        with th.inference_mode():
            # Generate description
            generated_ids = self.vqa_model.generate(
                **inputs,
                max_new_tokens=32, 
                do_sample=False
            )

            # Decode and parse answer
            outputs = generated_ids[0][inputs_len:]
            description = self.vqa_model_tokenizer.decode(outputs, skip_special_tokens=True) # pyright: ignore[ reportAttributeAccessIssue ]


        # Pray to the AI overloads that the description is not malformed or incorrect
        return description


    def _generate_prompts(self, description: str, label: str) -> list[str]:
        prompts = []


        while len(prompts) < self.num_variants:
            # Generate prompt
            # NOTE: You have no idea how long it took me to prompt engineer this
            instructions = [
                {"role": "system", "content": "You are strictly a text model part of an image generation pipeline. Your task is to slightly reformulate the text given by the user/program. You must provide MULTIPLE reformulations as a Python list of strings, i.e. ['reformulation 1', 'reformulation 2', ...]. You must ONLY answer with the provided format else you will break the image generation pipeline. You do NOT want to break the image generation pipeline. Change up the colors and lighting in the reformulations."},
                {"role": "user", "content": description},
            ]

            # Prompt model to give more descriptions
            outputs = self.prompt_model(instructions, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

            # Parse output
            output: str = cast(str, outputs[0]["generated_text"][-1]['content'].strip().strip("[]'\"")) # type: ignore
            prompts.extend(re.split(r"[\"\'],[ \n]*[\"\']", output))


        # Take the first num_variants, prepend the label and return the prompt
        return [f"The photo depicts: {label}. {prompt}" for prompt in prompts[:self.num_variants]]


    def _generate_inpaint(self, img: th.Tensor, mask: th.Tensor, depths: th.Tensor, edges: th.Tensor, prompt: str) -> th.Tensor:
        depth_min, depth_max = depths.min(), depths.max()
        depths = (depths - depth_min) / (depth_max - depth_min)
        edges = resize(edges, list(depths.shape))

        variant = self.diffusion_model(
            prompt=prompt,
            negative_prompt="low quality, bad quality, sketches, blurry, artifacts",
            num_inference_steps=20,
            eta=1.0,
            image=img,
            mask_image=mask,
            control_image=[depths[None, None].repeat(1, 3, 1, 1), edges[None]],
            strength=1.0,
            controlnet_conditioning_scale=[0.9, 0.5],
            guidance_scale=7.5,
        ).images[0] # type: ignore

        return resize(to_tensor(variant).to(self.device, dtype=self.dtype), [img.shape[1], img.shape[2]])


    def generate_variants(self, img_dir: Path) -> None:
        # Load image and annotations
        # TODO: Save everything as a dict of tensors/strings/lists to avoid weights_only=False
        img = Image.open(img_dir / "img.png")

        annotations = th.load(img_dir / "anno.pth", weights_only=False)
        h, w = annotations["img_shape"]
        bboxes = annotations["boxes"]
        masks = annotations["masks"]
        labels = annotations["labels"]

        # # Ensure that overlap of masks are accounted for 
        # new_mask = th.zeros((h,w), dtype=th.long)-1
        # idx_areas = th.tensor([mask.sum().item() for mask in masks]).argsort(descending=True)
        # for i in idx_areas: 
        #     new_mask[masks[i].bool()] = i
        
        # for i in range(len(masks)): 
        #     masks[i] = new_mask==i


        # Prepare output files
        output_img_dir = self.output_dir / img_dir.name
        output_img_dir.mkdir(parents=True, exist_ok=True)
        variant_dir = output_img_dir / "variants"
        variant_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = output_img_dir / "variants.txt"
        metadata_file.write_text("")


        # Estimate depth and edges for ControlNet
        depths = self._estimate_depth(img)
        edges = self._estimate_edges(img)
        img = to_tensor(img).to(self.device)


        # Generate variants for each instance
        for instance_idx in (pbar := trange(len(bboxes), leave=False)):
            # Bounding box for instance
            pbar.set_description(f"Instance - Setup")
            x1, y1, x2, y2 = bboxes[instance_idx]
            context_factor = 0.05

            # If instance is small, skip it
            if (x2 - x1) < self.bbox_min_side_length or (y2 - y1) < self.bbox_min_side_length:
                continue

            # Decode instance segmentation mask
            mask = th.tensor(decode(masks[instance_idx]), dtype=self.dtype).to(self.device)

            # Mean depth to determine draw order
            depth_instance = (depths[instance_idx] * mask).sum() / (mask.sum() + 1e-6)

            # Get initial base prompt
            # NOTE: Expanding the bounding box to give some context
            pbar.set_description(f"Instance - Describing")
            prompt_base = self._describe_object(
                img[
                    :,
                    int(y1*(1-context_factor)):int(y2*(1+context_factor)),
                    int(x1*(1-context_factor)):int(x2*(1+context_factor)),
                ]
            )

            # Reforumlate prompt into multiple variants
            pbar.set_description(f"Instance - Prompting")
            prompts = self._generate_prompts(prompt_base, index_to_name[labels[instance_idx].item()])


            # Generate variants
            pbar.set_description(f"Instance - Inpainting")
            for variant_idx in trange(self.num_variants, desc="Variant", leave=False):
                # Inpaint segmentation
                variant = self._generate_inpaint(img, mask, depths, edges, prompts[variant_idx])
                
                # Crop to bounding box and save variant
                crop = variant[:, y1:y2, x1:x2].to(device=th.device("cpu"), dtype=th.float32)
                variant_file = variant_dir / f"{instance_idx}_{variant_idx}.png"
                save_image(crop, str(variant_file))

                # Save metadata
                with open(metadata_file, 'a') as file:
                    file.write(f"{instance_idx} {variant_idx} {depth_instance.item()} {variant_dir.relative_to(self.output_dir)}\n")



    def run(self, start: int | float = 0.0, end: int | float = 1.0) -> None:
        if isinstance(start, float):
            start = int(start * len(self.img_dirs))
        if isinstance(end, float):
            end = int(end * len(self.img_dirs))

        for img_dir in (pbar := tqdm(self.img_dirs[start:end])):
            pbar.set_description(f"Processing - {img_dir}")
            self.generate_variants(self.input_dir / img_dir)



if __name__ == "__main__":
    args = tyro.cli(Config)
    VariantGeneration(
        input_dir=Path(args.dataloader.processed_data_dir) / "train",
        num_variants=args.varient_generation.num_variants,
        bbox_min_side_length=args.varient_generation.bbox_min_side_length
    ).run(start=args.varient_generation.subset_start, end=args.varient_generation.subset_end)

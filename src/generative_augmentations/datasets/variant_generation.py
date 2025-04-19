from typing import cast
from pathlib import Path
from PIL import Image
import re
import logging

import torch as th
from torchvision.transforms.functional import to_tensor
from joblib import Parallel, delayed
from tqdm import tqdm
from pycocotools.mask import decode
from transformers.pipelines import pipeline
from transformers.pipelines.depth_estimation import DepthEstimationPipeline
from transformers.pipelines.text_generation import TextGenerationPipeline
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from controlnet_aux.processor import HEDdetector


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



class VariantGeneration:
    def __init__(self, input_dir: Path, output_dir: Path | None = None, num_variants: int = 3, num_workers: int = -4, device: th.device | None = None) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir if output_dir else input_dir
        self.num_variants = num_variants

        self.num_workers = num_workers
        self.device = device if device else th.device("cuda" if th.cuda.is_available() else "cpu")
        self.dtype = th.float16 if th.cuda.is_available() else th.float32
        # Based upon: https://arxiv.org/abs/2406.09414
        self.depth_model = self._get_depth_model()
        # Based upon: https://arxiv.org/abs/2302.05543
        self.edge_model = HEDdetector.from_pretrained("lllyasviel/Annotators").to(self.device)
        # Based upon: https://arxiv.org/abs/2311.06242
        self.vqa_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=self.dtype, trust_remote_code=True).to(device)
        self.vqa_model_tokenizer = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        # Based upon:
        self.prompt_model = self._get_prompt_model()
        

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
            pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", torch_dtype=self.dtype, device_map=self.device)
        )

        # Reset logger level
        pipe_logger.setLevel(pipe_logger_level_orig)

        # Return pipeline
        return pipe


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



    def _estimate_depth(self, img: Image.Image) -> th.Tensor:
        return self.depth_model(img)["predicted_depth"].to(self.device) # type: ignore


    def _estimate_edges(self, img: Image.Image) -> th.Tensor:
        return th.tensor(self.edge_model(img, output_type="numpy")).to(device=self.device, dtype=th.float32) / 255.0


    def _describe_object(self, img: th.Tensor) -> str:
        # NOTE: If the bounding box contains multiple objects,
        #  this method of obtaining a description will cause issues.
        #  In such a case, maybe cut out the instance via the mask first.
        
        # Prepare prompt
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.vqa_model_tokenizer(
            text=prompt,
            images=img,
            return_tensors="pt"
        ).to(self.vqa_model.device, self.vqa_model.dtype)

        # Generate description
        generated_ids = self.vqa_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=24,
            num_beams=3
        )

        # Decode and parse answer
        outputs = self.vqa_model_tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        description = self.vqa_model_tokenizer.post_process_generation(
            outputs,
            task=prompt,
            image_size=img.shape[1:]
        )["<MORE_DETAILED_CAPTION>"]


        # Pray to the AI overloads that the description is not malformed or incorrect
        return description


    def _generate_prompts(self, description: str, label: str) -> list[str]:
        prompts = []


        while len(prompts) < self.num_variants:
            # Generate prompt
            # NOTE: You have no idea how long it took me to prompt engineer this
            instructions = [
                {"role": "system", "content": "You are strictly a text model part of an image generation pipeline. Your task is to slightly reformulate the text given by the user/program. You must provide MULTIPLE reformulations as a Python list of strings. You must ONLY answer with the provided format else you will break the image generation pipeline. You do NOT want to break the image generation pipeline. Change up the colors and lighting in the reformulations."},
                {"role": "user", "content": "The image is a close-up of a yellow tennis ball in the air. The ball is in focus,"},
            ]

            # Prompt model to give more descriptions
            outputs = self.prompt_model(instructions, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

            # Parse output
            output: str = cast(str, outputs[0]["generated_text"][-1]['content'].strip().strip("[]'\"")) # type: ignore
            prompts.extend(re.split(r"[\"\'], ?[\"\']", output))


        # Take the first num_variants, prepend the label and return the prompt
        return [f"The image is of a {label}. {prompt}" for prompt in prompts[:self.num_variants]]


    def generate_variants(self, img_dir: Path) -> None:
        # TODO: Save everything as a dict of tensors/strings/lists to avoid weights_only=False
        annotations = th.load(img_dir / "anno.pth", weights_only=False)
        h, w = annotations["img_shape"]
        bboxes = annotations["boxes"]
        masks = annotations["masks"]
        labels = annotations["labels"]
        


        img = Image.open(img_dir / "img.png")
        depths = self._estimate_depth(img)
        edges = self._estimate_edges(img)
        img = to_tensor(img).to(self.device)

        for instance_idx in range(len(bboxes)):
            # Decode mask
            mask = decode(masks[instance_idx])


            # TODO: Estimate mean depth for this instance to determine draw order


            x1, y1, x2, y2 = bboxes[instance_idx]
            context_factor = 0.05

            # Get initial base prompt
            # NOTE: Expanding the bounding box to give some context
            prompt_base = self._describe_object(
                img[
                    :,
                    int(y1*(1-context_factor)):int(y2*(1+context_factor)),
                    int(x1*(1-context_factor)):int(x2*(1+context_factor)),
                ] * 255
            )

            # Reforumlate prompt into multiple variants
            # TODO: Map label index to label name
            prompts = self._generate_prompts(prompt_base, labels[instance_idx])



            for variant_idx in range(self.num_variants):
                # Inpaint image based upon mask, depth, edge, and prompt
                
                


    def run(self) -> None:
        # TODO: Run stuff in parallel, process all images
        # TODO: Pretty progress bars


if __name__ == "__main__":
    ...
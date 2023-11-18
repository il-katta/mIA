Help on StableDiffusionXLPipeline in module diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl object:

class StableDiffusionXLPipeline(diffusers.pipelines.pipeline_utils.DiffusionPipeline, diffusers.loaders.FromSingleFileMixin, diffusers.loaders.LoraLoaderMixin)
 |  StableDiffusionXLPipeline(vae: diffusers.models.autoencoder_kl.AutoencoderKL, text_encoder: transformers.models.clip.modeling_clip.CLIPTextModel, text_encoder_2: transformers.models.clip.modeling_clip.CLIPTextModelWithProjection, tokenizer: transformers.models.clip.tokenization_clip.CLIPTokenizer, tokenizer_2: transformers.models.clip.tokenization_clip.CLIPTokenizer, unet: diffusers.models.unet_2d_condition.UNet2DConditionModel, scheduler: diffusers.schedulers.scheduling_utils.KarrasDiffusionSchedulers, force_zeros_for_empty_prompt: bool = True, add_watermarker: Optional[bool] = None)
 |  
 |  Pipeline for text-to-image generation using Stable Diffusion XL.
 |  
 |  This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
 |  library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
 |  
 |  In addition the pipeline inherits the following loading methods:
 |      - *Textual-Inversion*: [`loaders.TextualInversionLoaderMixin.load_textual_inversion`]
 |      - *LoRA*: [`StableDiffusionXLPipeline.load_lora_weights`]
 |      - *Ckpt*: [`loaders.FromSingleFileMixin.from_single_file`]
 |  
 |  as well as the following saving methods:
 |      - *LoRA*: [`loaders.StableDiffusionXLPipeline.save_lora_weights`]
 |  
 |  Args:
 |      vae ([`AutoencoderKL`]):
 |          Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
 |      text_encoder ([`CLIPTextModel`]):
 |          Frozen text-encoder. Stable Diffusion XL uses the text portion of
 |          [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
 |          the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
 |      text_encoder_2 ([` CLIPTextModelWithProjection`]):
 |          Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
 |          [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
 |          specifically the
 |          [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
 |          variant.
 |      tokenizer (`CLIPTokenizer`):
 |          Tokenizer of class
 |          [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
 |      tokenizer_2 (`CLIPTokenizer`):
 |          Second Tokenizer of class
 |          [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
 |      unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
 |      scheduler ([`SchedulerMixin`]):
 |          A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
 |          [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
 |  
 |  Method resolution order:
 |      StableDiffusionXLPipeline
 |      diffusers.pipelines.pipeline_utils.DiffusionPipeline
 |      diffusers.configuration_utils.ConfigMixin
 |      diffusers.loaders.FromSingleFileMixin
 |      diffusers.loaders.LoraLoaderMixin
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __call__(self, prompt: Union[str, List[str]] = None, prompt_2: Union[str, List[str], NoneType] = None, height: Optional[int] = None, width: Optional[int] = None, num_inference_steps: int = 50, denoising_end: Optional[float] = None, guidance_scale: float = 5.0, negative_prompt: Union[str, List[str], NoneType] = None, negative_prompt_2: Union[str, List[str], NoneType] = None, num_images_per_prompt: Optional[int] = 1, eta: float = 0.0, generator: Union[torch._C.Generator, List[torch._C.Generator], NoneType] = None, latents: Optional[torch.FloatTensor] = None, prompt_embeds: Optional[torch.FloatTensor] = None, negative_prompt_embeds: Optional[torch.FloatTensor] = None, pooled_prompt_embeds: Optional[torch.FloatTensor] = None, negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None, output_type: Optional[str] = 'pil', return_dict: bool = True, callback: Optional[Callable[[int, int, torch.FloatTensor], NoneType]] = None, callback_steps: int = 1, cross_attention_kwargs: Optional[Dict[str, Any]] = None, guidance_rescale: float = 0.0, original_size: Optional[Tuple[int, int]] = None, crops_coords_top_left: Tuple[int, int] = (0, 0), target_size: Optional[Tuple[int, int]] = None)
 |          Function invoked when calling the pipeline for generation.
 |      
 |          Args:
 |              prompt (`str` or `List[str]`, *optional*):
 |                  The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
 |                  instead.
 |              prompt_2 (`str` or `List[str]`, *optional*):
 |                  The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
 |                  used in both text-encoders
 |              height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
 |                  The height in pixels of the generated image.
 |              width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
 |                  The width in pixels of the generated image.
 |              num_inference_steps (`int`, *optional*, defaults to 50):
 |                  The number of denoising steps. More denoising steps usually lead to a higher quality image at the
 |                  expense of slower inference.
 |              denoising_end (`float`, *optional*):
 |                  When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
 |                  completed before it is intentionally prematurely terminated. As a result, the returned sample will
 |                  still retain a substantial amount of noise as determined by the discrete timesteps selected by the
 |                  scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
 |                  "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
 |                  Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
 |              guidance_scale (`float`, *optional*, defaults to 7.5):
 |                  Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
 |                  `guidance_scale` is defined as `w` of equation 2. of [Imagen
 |                  Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
 |                  1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
 |                  usually at the expense of lower image quality.
 |              negative_prompt (`str` or `List[str]`, *optional*):
 |                  The prompt or prompts not to guide the image generation. If not defined, one has to pass
 |                  `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
 |                  less than `1`).
 |              negative_prompt_2 (`str` or `List[str]`, *optional*):
 |                  The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
 |                  `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
 |              num_images_per_prompt (`int`, *optional*, defaults to 1):
 |                  The number of images to generate per prompt.
 |              eta (`float`, *optional*, defaults to 0.0):
 |                  Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
 |                  [`schedulers.DDIMScheduler`], will be ignored for others.
 |              generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
 |                  One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
 |                  to make generation deterministic.
 |              latents (`torch.FloatTensor`, *optional*):
 |                  Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
 |                  generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
 |                  tensor will ge generated by sampling using the supplied random `generator`.
 |              prompt_embeds (`torch.FloatTensor`, *optional*):
 |                  Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
 |                  provided, text embeddings will be generated from `prompt` input argument.
 |              negative_prompt_embeds (`torch.FloatTensor`, *optional*):
 |                  Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
 |                  weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
 |                  argument.
 |              pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
 |                  Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
 |                  If not provided, pooled text embeddings will be generated from `prompt` input argument.
 |              negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
 |                  Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
 |                  weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
 |                  input argument.
 |              output_type (`str`, *optional*, defaults to `"pil"`):
 |                  The output format of the generate image. Choose between
 |                  [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
 |              return_dict (`bool`, *optional*, defaults to `True`):
 |                  Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
 |                  of a plain tuple.
 |              callback (`Callable`, *optional*):
 |                  A function that will be called every `callback_steps` steps during inference. The function will be
 |                  called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
 |              callback_steps (`int`, *optional*, defaults to 1):
 |                  The frequency at which the `callback` function will be called. If not specified, the callback will be
 |                  called at every step.
 |              cross_attention_kwargs (`dict`, *optional*):
 |                  A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
 |                  `self.processor` in
 |                  [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
 |              guidance_rescale (`float`, *optional*, defaults to 0.7):
 |                  Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
 |                  Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
 |                  [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
 |                  Guidance rescale factor should fix overexposure when using zero terminal SNR.
 |              original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
 |                  If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
 |                  `original_size` defaults to `(width, height)` if not specified. Part of SDXL's micro-conditioning as
 |                  explained in section 2.2 of
 |                  [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
 |              crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
 |                  `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
 |                  `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
 |                  `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
 |                  [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
 |              target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
 |                  For most cases, `target_size` should be set to the desired height and width of the generated image. If
 |                  not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
 |                  section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
 |      
 |      
 |      Examples:
 |          ```py
 |          >>> import torch
 |          >>> from diffusers import StableDiffusionXLPipeline
 |      
 |          >>> pipe = StableDiffusionXLPipeline.from_pretrained(
 |          ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
 |          ... )
 |          >>> pipe = pipe.to("cuda")
 |      
 |          >>> prompt = "a photo of an astronaut riding a horse on mars"
 |          >>> image = pipe(prompt).images[0]
 |          ```
 |      
 |      
 |          Returns:
 |              [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
 |              [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
 |              `tuple`. When returning a tuple, the first element is a list with the generated images.
 |  
 |  __init__(self, vae: diffusers.models.autoencoder_kl.AutoencoderKL, text_encoder: transformers.models.clip.modeling_clip.CLIPTextModel, text_encoder_2: transformers.models.clip.modeling_clip.CLIPTextModelWithProjection, tokenizer: transformers.models.clip.tokenization_clip.CLIPTokenizer, tokenizer_2: transformers.models.clip.tokenization_clip.CLIPTokenizer, unet: diffusers.models.unet_2d_condition.UNet2DConditionModel, scheduler: diffusers.schedulers.scheduling_utils.KarrasDiffusionSchedulers, force_zeros_for_empty_prompt: bool = True, add_watermarker: Optional[bool] = None)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  check_inputs(self, prompt, prompt_2, height, width, callback_steps, negative_prompt=None, negative_prompt_2=None, prompt_embeds=None, negative_prompt_embeds=None, pooled_prompt_embeds=None, negative_pooled_prompt_embeds=None)
 |  
 |  disable_vae_slicing(self)
 |      Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
 |      computing decoding in one step.
 |  
 |  disable_vae_tiling(self)
 |      Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
 |      computing decoding in one step.
 |  
 |  enable_model_cpu_offload(self, gpu_id=0)
 |      Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
 |      to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
 |      method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
 |      `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
 |  
 |  enable_vae_slicing(self)
 |      Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
 |      compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
 |  
 |  enable_vae_tiling(self)
 |      Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
 |      compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
 |      processing larger images.
 |  
 |  encode_prompt(self, prompt: str, prompt_2: Optional[str] = None, device: Optional[torch.device] = None, num_images_per_prompt: int = 1, do_classifier_free_guidance: bool = True, negative_prompt: Optional[str] = None, negative_prompt_2: Optional[str] = None, prompt_embeds: Optional[torch.FloatTensor] = None, negative_prompt_embeds: Optional[torch.FloatTensor] = None, pooled_prompt_embeds: Optional[torch.FloatTensor] = None, negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None, lora_scale: Optional[float] = None)
 |      Encodes the prompt into text encoder hidden states.
 |      
 |      Args:
 |          prompt (`str` or `List[str]`, *optional*):
 |              prompt to be encoded
 |          prompt_2 (`str` or `List[str]`, *optional*):
 |              The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
 |              used in both text-encoders
 |          device: (`torch.device`):
 |              torch device
 |          num_images_per_prompt (`int`):
 |              number of images that should be generated per prompt
 |          do_classifier_free_guidance (`bool`):
 |              whether to use classifier free guidance or not
 |          negative_prompt (`str` or `List[str]`, *optional*):
 |              The prompt or prompts not to guide the image generation. If not defined, one has to pass
 |              `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
 |              less than `1`).
 |          negative_prompt_2 (`str` or `List[str]`, *optional*):
 |              The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
 |              `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
 |          prompt_embeds (`torch.FloatTensor`, *optional*):
 |              Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
 |              provided, text embeddings will be generated from `prompt` input argument.
 |          negative_prompt_embeds (`torch.FloatTensor`, *optional*):
 |              Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
 |              weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
 |              argument.
 |          pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
 |              Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
 |              If not provided, pooled text embeddings will be generated from `prompt` input argument.
 |          negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
 |              Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
 |              weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
 |              input argument.
 |          lora_scale (`float`, *optional*):
 |              A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
 |  
 |  load_lora_weights(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs)
 |      Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.unet` and
 |      `self.text_encoder`.
 |      
 |      All kwargs are forwarded to `self.lora_state_dict`.
 |      
 |      See [`~loaders.LoraLoaderMixin.lora_state_dict`] for more details on how the state dict is loaded.
 |      
 |      See [`~loaders.LoraLoaderMixin.load_lora_into_unet`] for more details on how the state dict is loaded into
 |      `self.unet`.
 |      
 |      See [`~loaders.LoraLoaderMixin.load_lora_into_text_encoder`] for more details on how the state dict is loaded
 |      into `self.text_encoder`.
 |      
 |      Parameters:
 |          pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
 |              See [`~loaders.LoraLoaderMixin.lora_state_dict`].
 |          kwargs (`dict`, *optional*):
 |              See [`~loaders.LoraLoaderMixin.lora_state_dict`].
 |  
 |  prepare_extra_step_kwargs(self, generator, eta)
 |      # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
 |  
 |  prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None)
 |      # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
 |  
 |  upcast_vae(self)
 |      # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
 |  
 |  ----------------------------------------------------------------------
 |  Class methods defined here:
 |  
 |  save_lora_weights(save_directory: Union[str, os.PathLike], unet_lora_layers: Dict[str, Union[torch.nn.modules.module.Module, torch.Tensor]] = None, text_encoder_lora_layers: Dict[str, Union[torch.nn.modules.module.Module, torch.Tensor]] = None, text_encoder_2_lora_layers: Dict[str, Union[torch.nn.modules.module.Module, torch.Tensor]] = None, is_main_process: bool = True, weight_name: str = None, save_function: Callable = None, safe_serialization: bool = False) from builtins.type
 |      Save the LoRA parameters corresponding to the UNet and text encoder.
 |      
 |      Arguments:
 |          save_directory (`str` or `os.PathLike`):
 |              Directory to save LoRA parameters to. Will be created if it doesn't exist.
 |          unet_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
 |              State dict of the LoRA layers corresponding to the `unet`.
 |          text_encoder_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
 |              State dict of the LoRA layers corresponding to the `text_encoder`. Must explicitly pass the text
 |              encoder LoRA state dict because it comes from 🤗 Transformers.
 |          is_main_process (`bool`, *optional*, defaults to `True`):
 |              Whether the process calling this is the main process or not. Useful during distributed training and you
 |              need to call this function on all processes. In this case, set `is_main_process=True` only on the main
 |              process to avoid race conditions.
 |          save_function (`Callable`):
 |              The function to use to save the state dictionary. Useful during distributed training when you need to
 |              replace `torch.save` with another method. Can be configured with the environment variable
 |              `DIFFUSERS_SAVE_MODE`.
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from diffusers.pipelines.pipeline_utils.DiffusionPipeline:
 |  
 |  __setattr__(self, name: str, value: Any)
 |      Implement setattr(self, name, value).
 |  
 |  disable_attention_slicing(self)
 |      Disable sliced attention computation. If `enable_attention_slicing` was previously called, attention is
 |      computed in one step.
 |  
 |  disable_xformers_memory_efficient_attention(self)
 |      Disable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/).
 |  
 |  enable_attention_slicing(self, slice_size: Union[str, int, NoneType] = 'auto')
 |      Enable sliced attention computation. When this option is enabled, the attention module splits the input tensor
 |      in slices to compute attention in several steps. This is useful to save some memory in exchange for a small
 |      speed decrease.
 |      
 |      Args:
 |          slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
 |              When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
 |              `"max"`, maximum amount of memory will be saved by running only one slice at a time. If a number is
 |              provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
 |              must be a multiple of `slice_size`.
 |  
 |  enable_sequential_cpu_offload(self, gpu_id: int = 0, device: Union[torch.device, str] = 'cuda')
 |      Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
 |      text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
 |      `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
 |      Note that offloading happens on a submodule basis. Memory savings are higher than with
 |      `enable_model_cpu_offload`, but performance is lower.
 |  
 |  enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None)
 |      Enable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/). When this
 |      option is enabled, you should observe lower GPU memory usage and a potential speed up during inference. Speed
 |      up during training is not guaranteed.
 |      
 |      <Tip warning={true}>
 |      
 |      ⚠️ When memory efficient attention and sliced attention are both enabled, memory efficient attention takes
 |      precedent.
 |      
 |      </Tip>
 |      
 |      Parameters:
 |          attention_op (`Callable`, *optional*):
 |              Override the default `None` operator for use as `op` argument to the
 |              [`memory_efficient_attention()`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)
 |              function of xFormers.
 |      
 |      Examples:
 |      
 |      ```py
 |      >>> import torch
 |      >>> from diffusers import DiffusionPipeline
 |      >>> from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
 |      
 |      >>> pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
 |      >>> pipe = pipe.to("cuda")
 |      >>> pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
 |      >>> # Workaround for not accepting attention shape using VAE for Flash Attention
 |      >>> pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
 |      ```
 |  
 |  progress_bar(self, iterable=None, total=None)
 |  
 |  register_modules(self, **kwargs)
 |  
 |  save_pretrained(self, save_directory: Union[str, os.PathLike], safe_serialization: bool = False, variant: Optional[str] = None)
 |      Save all saveable variables of the pipeline to a directory. A pipeline variable can be saved and loaded if its
 |      class implements both a save and loading method. The pipeline is easily reloaded using the
 |      [`~DiffusionPipeline.from_pretrained`] class method.
 |      
 |      Arguments:
 |          save_directory (`str` or `os.PathLike`):
 |              Directory to save a pipeline to. Will be created if it doesn't exist.
 |          safe_serialization (`bool`, *optional*, defaults to `False`):
 |              Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
 |          variant (`str`, *optional*):
 |              If specified, weights are saved in the format `pytorch_model.<variant>.bin`.
 |  
 |  set_attention_slice(self, slice_size: Optional[int])
 |  
 |  set_progress_bar_config(self, **kwargs)
 |  
 |  set_use_memory_efficient_attention_xformers(self, valid: bool, attention_op: Optional[Callable] = None) -> None
 |  
 |  to(self, torch_device: Union[str, torch.device, NoneType] = None, torch_dtype: Optional[torch.dtype] = None, silence_dtype_warnings: bool = False)
 |  
 |  ----------------------------------------------------------------------
 |  Class methods inherited from diffusers.pipelines.pipeline_utils.DiffusionPipeline:
 |  
 |  download(pretrained_model_name, **kwargs) -> Union[str, os.PathLike] from builtins.type
 |      Download and cache a PyTorch diffusion pipeline from pretrained pipeline weights.
 |      
 |      Parameters:
 |          pretrained_model_name (`str` or `os.PathLike`, *optional*):
 |              A string, the *repository id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
 |              hosted on the Hub.
 |          custom_pipeline (`str`, *optional*):
 |              Can be either:
 |      
 |                  - A string, the *repository id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained
 |                    pipeline hosted on the Hub. The repository must contain a file called `pipeline.py` that defines
 |                    the custom pipeline.
 |      
 |                  - A string, the *file name* of a community pipeline hosted on GitHub under
 |                    [Community](https://github.com/huggingface/diffusers/tree/main/examples/community). Valid file
 |                    names must match the file name and not the pipeline script (`clip_guided_stable_diffusion`
 |                    instead of `clip_guided_stable_diffusion.py`). Community pipelines are always loaded from the
 |                    current `main` branch of GitHub.
 |      
 |                  - A path to a *directory* (`./my_pipeline_directory/`) containing a custom pipeline. The directory
 |                    must contain a file called `pipeline.py` that defines the custom pipeline.
 |      
 |              <Tip warning={true}>
 |      
 |              🧪 This is an experimental feature and may change in the future.
 |      
 |              </Tip>
 |      
 |              For more information on how to load and create custom pipelines, take a look at [How to contribute a
 |              community pipeline](https://huggingface.co/docs/diffusers/main/en/using-diffusers/contribute_pipeline).
 |      
 |          force_download (`bool`, *optional*, defaults to `False`):
 |              Whether or not to force the (re-)download of the model weights and configuration files, overriding the
 |              cached versions if they exist.
 |          resume_download (`bool`, *optional*, defaults to `False`):
 |              Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
 |              incompletely downloaded files are deleted.
 |          proxies (`Dict[str, str]`, *optional*):
 |              A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
 |              'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
 |          output_loading_info(`bool`, *optional*, defaults to `False`):
 |              Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
 |          local_files_only (`bool`, *optional*, defaults to `False`):
 |              Whether to only load local model weights and configuration files or not. If set to `True`, the model
 |              won't be downloaded from the Hub.
 |          use_auth_token (`str` or *bool*, *optional*):
 |              The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
 |              `diffusers-cli login` (stored in `~/.huggingface`) is used.
 |          revision (`str`, *optional*, defaults to `"main"`):
 |              The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
 |              allowed by Git.
 |          custom_revision (`str`, *optional*, defaults to `"main"`):
 |              The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
 |              `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
 |              custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
 |          mirror (`str`, *optional*):
 |              Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
 |              guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
 |              information.
 |          variant (`str`, *optional*):
 |              Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
 |              loading `from_flax`.
 |          use_safetensors (`bool`, *optional*, defaults to `None`):
 |              If set to `None`, the safetensors weights are downloaded if they're available **and** if the
 |              safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
 |              weights. If set to `False`, safetensors weights are not loaded.
 |          use_onnx (`bool`, *optional*, defaults to `False`):
 |              If set to `True`, ONNX weights will always be downloaded if present. If set to `False`, ONNX weights
 |              will never be downloaded. By default `use_onnx` defaults to the `_is_onnx` class attribute which is
 |              `False` for non-ONNX pipelines and `True` for ONNX pipelines. ONNX weights include both files ending
 |              with `.onnx` and `.pb`.
 |      
 |      Returns:
 |          `os.PathLike`:
 |              A path to the downloaded pipeline.
 |      
 |      <Tip>
 |      
 |      To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with
 |      `huggingface-cli login`.
 |      
 |      </Tip>
 |  
 |  from_pretrained(pretrained_model_name_or_path: Union[str, os.PathLike, NoneType], **kwargs) from builtins.type
 |      Instantiate a PyTorch diffusion pipeline from pretrained pipeline weights.
 |      
 |      The pipeline is set in evaluation mode (`model.eval()`) by default.
 |      
 |      If you get the error message below, you need to finetune the weights for your downstream task:
 |      
 |      ```
 |      Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
 |      - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
 |      You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
 |      ```
 |      
 |      Parameters:
 |          pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
 |              Can be either:
 |      
 |                  - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
 |                    hosted on the Hub.
 |                  - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
 |                    saved using
 |                  [`~DiffusionPipeline.save_pretrained`].
 |          torch_dtype (`str` or `torch.dtype`, *optional*):
 |              Override the default `torch.dtype` and load the model with another dtype. If "auto" is passed, the
 |              dtype is automatically derived from the model's weights.
 |          custom_pipeline (`str`, *optional*):
 |      
 |              <Tip warning={true}>
 |      
 |              🧪 This is an experimental feature and may change in the future.
 |      
 |              </Tip>
 |      
 |              Can be either:
 |      
 |                  - A string, the *repo id* (for example `hf-internal-testing/diffusers-dummy-pipeline`) of a custom
 |                    pipeline hosted on the Hub. The repository must contain a file called pipeline.py that defines
 |                    the custom pipeline.
 |                  - A string, the *file name* of a community pipeline hosted on GitHub under
 |                    [Community](https://github.com/huggingface/diffusers/tree/main/examples/community). Valid file
 |                    names must match the file name and not the pipeline script (`clip_guided_stable_diffusion`
 |                    instead of `clip_guided_stable_diffusion.py`). Community pipelines are always loaded from the
 |                    current main branch of GitHub.
 |                  - A path to a directory (`./my_pipeline_directory/`) containing a custom pipeline. The directory
 |                    must contain a file called `pipeline.py` that defines the custom pipeline.
 |      
 |              For more information on how to load and create custom pipelines, please have a look at [Loading and
 |              Adding Custom
 |              Pipelines](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview)
 |          force_download (`bool`, *optional*, defaults to `False`):
 |              Whether or not to force the (re-)download of the model weights and configuration files, overriding the
 |              cached versions if they exist.
 |          cache_dir (`Union[str, os.PathLike]`, *optional*):
 |              Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
 |              is not used.
 |          resume_download (`bool`, *optional*, defaults to `False`):
 |              Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
 |              incompletely downloaded files are deleted.
 |          proxies (`Dict[str, str]`, *optional*):
 |              A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
 |              'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
 |          output_loading_info(`bool`, *optional*, defaults to `False`):
 |              Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
 |          local_files_only (`bool`, *optional*, defaults to `False`):
 |              Whether to only load local model weights and configuration files or not. If set to `True`, the model
 |              won't be downloaded from the Hub.
 |          use_auth_token (`str` or *bool*, *optional*):
 |              The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
 |              `diffusers-cli login` (stored in `~/.huggingface`) is used.
 |          revision (`str`, *optional*, defaults to `"main"`):
 |              The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
 |              allowed by Git.
 |          custom_revision (`str`, *optional*, defaults to `"main"`):
 |              The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
 |              `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
 |              custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
 |          mirror (`str`, *optional*):
 |              Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
 |              guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
 |              information.
 |          device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
 |              A map that specifies where each submodule should go. It doesn’t need to be defined for each
 |              parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
 |              same device.
 |      
 |              Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
 |              more information about each option see [designing a device
 |              map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
 |          max_memory (`Dict`, *optional*):
 |              A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
 |              each GPU and the available CPU RAM if unset.
 |          offload_folder (`str` or `os.PathLike`, *optional*):
 |              The path to offload weights if device_map contains the value `"disk"`.
 |          offload_state_dict (`bool`, *optional*):
 |              If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
 |              the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
 |              when there is some disk offload.
 |          low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
 |              Speed up model loading only loading the pretrained weights and not initializing the weights. This also
 |              tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
 |              Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
 |              argument to `True` will raise an error.
 |          use_safetensors (`bool`, *optional*, defaults to `None`):
 |              If set to `None`, the safetensors weights are downloaded if they're available **and** if the
 |              safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
 |              weights. If set to `False`, safetensors weights are not loaded.
 |          use_onnx (`bool`, *optional*, defaults to `None`):
 |              If set to `True`, ONNX weights will always be downloaded if present. If set to `False`, ONNX weights
 |              will never be downloaded. By default `use_onnx` defaults to the `_is_onnx` class attribute which is
 |              `False` for non-ONNX pipelines and `True` for ONNX pipelines. ONNX weights include both files ending
 |              with `.onnx` and `.pb`.
 |          kwargs (remaining dictionary of keyword arguments, *optional*):
 |              Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
 |              class). The overwritten components are passed directly to the pipelines `__init__` method. See example
 |              below for more information.
 |          variant (`str`, *optional*):
 |              Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
 |              loading `from_flax`.
 |      
 |      <Tip>
 |      
 |      To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
 |      `huggingface-cli login`.
 |      
 |      </Tip>
 |      
 |      Examples:
 |      
 |      ```py
 |      >>> from diffusers import DiffusionPipeline
 |      
 |      >>> # Download pipeline from huggingface.co and cache.
 |      >>> pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")
 |      
 |      >>> # Download pipeline that requires an authorization token
 |      >>> # For more information on access tokens, please refer to this section
 |      >>> # of the documentation](https://huggingface.co/docs/hub/security-tokens)
 |      >>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
 |      
 |      >>> # Use a different scheduler
 |      >>> from diffusers import LMSDiscreteScheduler
 |      
 |      >>> scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
 |      >>> pipeline.scheduler = scheduler
 |      ```
 |  
 |  ----------------------------------------------------------------------
 |  Static methods inherited from diffusers.pipelines.pipeline_utils.DiffusionPipeline:
 |  
 |  numpy_to_pil(images)
 |      Convert a NumPy image or a batch of images to a PIL image.
 |  
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from diffusers.pipelines.pipeline_utils.DiffusionPipeline:
 |  
 |  components
 |      The `self.components` property can be useful to run different pipelines with the same weights and
 |      configurations without reallocating additional memory.
 |      
 |      Returns (`dict`):
 |          A dictionary containing all the modules needed to initialize the pipeline.
 |      
 |      Examples:
 |      
 |      ```py
 |      >>> from diffusers import (
 |      ...     StableDiffusionPipeline,
 |      ...     StableDiffusionImg2ImgPipeline,
 |      ...     StableDiffusionInpaintPipeline,
 |      ... )
 |      
 |      >>> text2img = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
 |      >>> img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
 |      >>> inpaint = StableDiffusionInpaintPipeline(**text2img.components)
 |      ```
 |  
 |  device
 |      Returns:
 |          `torch.device`: The torch device on which the pipeline is located.
 |  
 |  name_or_path
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from diffusers.pipelines.pipeline_utils.DiffusionPipeline:
 |  
 |  config_name = 'model_index.json'
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from diffusers.configuration_utils.ConfigMixin:
 |  
 |  __getattr__(self, name: str) -> Any
 |      The only reason we overwrite `getattr` here is to gracefully deprecate accessing
 |      config attributes directly. See https://github.com/huggingface/diffusers/pull/3129
 |      
 |      Tihs funtion is mostly copied from PyTorch's __getattr__ overwrite:
 |      https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
 |  
 |  __repr__(self)
 |      Return repr(self).
 |  
 |  register_to_config(self, **kwargs)
 |  
 |  save_config(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs)
 |      Save a configuration object to the directory specified in `save_directory` so that it can be reloaded using the
 |      [`~ConfigMixin.from_config`] class method.
 |      
 |      Args:
 |          save_directory (`str` or `os.PathLike`):
 |              Directory where the configuration JSON file is saved (will be created if it does not exist).
 |  
 |  to_json_file(self, json_file_path: Union[str, os.PathLike])
 |      Save the configuration instance's parameters to a JSON file.
 |      
 |      Args:
 |          json_file_path (`str` or `os.PathLike`):
 |              Path to the JSON file to save a configuration instance's parameters.
 |  
 |  to_json_string(self) -> str
 |      Serializes the configuration instance to a JSON string.
 |      
 |      Returns:
 |          `str`:
 |              String containing all the attributes that make up the configuration instance in JSON format.
 |  
 |  ----------------------------------------------------------------------
 |  Class methods inherited from diffusers.configuration_utils.ConfigMixin:
 |  
 |  extract_init_dict(config_dict, **kwargs) from builtins.type
 |  
 |  from_config(config: Union[diffusers.configuration_utils.FrozenDict, Dict[str, Any]] = None, return_unused_kwargs=False, **kwargs) from builtins.type
 |      Instantiate a Python class from a config dictionary.
 |      
 |      Parameters:
 |          config (`Dict[str, Any]`):
 |              A config dictionary from which the Python class is instantiated. Make sure to only load configuration
 |              files of compatible classes.
 |          return_unused_kwargs (`bool`, *optional*, defaults to `False`):
 |              Whether kwargs that are not consumed by the Python class should be returned or not.
 |          kwargs (remaining dictionary of keyword arguments, *optional*):
 |              Can be used to update the configuration object (after it is loaded) and initiate the Python class.
 |              `**kwargs` are passed directly to the underlying scheduler/model's `__init__` method and eventually
 |              overwrite the same named arguments in `config`.
 |      
 |      Returns:
 |          [`ModelMixin`] or [`SchedulerMixin`]:
 |              A model or scheduler object instantiated from a config dictionary.
 |      
 |      Examples:
 |      
 |      ```python
 |      >>> from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler
 |      
 |      >>> # Download scheduler from huggingface.co and cache.
 |      >>> scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")
 |      
 |      >>> # Instantiate DDIM scheduler class with same config as DDPM
 |      >>> scheduler = DDIMScheduler.from_config(scheduler.config)
 |      
 |      >>> # Instantiate PNDM scheduler class with same config as DDPM
 |      >>> scheduler = PNDMScheduler.from_config(scheduler.config)
 |      ```
 |  
 |  get_config_dict(*args, **kwargs) from builtins.type
 |  
 |  load_config(pretrained_model_name_or_path: Union[str, os.PathLike], return_unused_kwargs=False, return_commit_hash=False, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]] from builtins.type
 |      Load a model or scheduler configuration.
 |      
 |      Parameters:
 |          pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
 |              Can be either:
 |      
 |                  - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
 |                    the Hub.
 |                  - A path to a *directory* (for example `./my_model_directory`) containing model weights saved with
 |                    [`~ConfigMixin.save_config`].
 |      
 |          cache_dir (`Union[str, os.PathLike]`, *optional*):
 |              Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
 |              is not used.
 |          force_download (`bool`, *optional*, defaults to `False`):
 |              Whether or not to force the (re-)download of the model weights and configuration files, overriding the
 |              cached versions if they exist.
 |          resume_download (`bool`, *optional*, defaults to `False`):
 |              Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
 |              incompletely downloaded files are deleted.
 |          proxies (`Dict[str, str]`, *optional*):
 |              A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
 |              'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
 |          output_loading_info(`bool`, *optional*, defaults to `False`):
 |              Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
 |          local_files_only (`bool`, *optional*, defaults to `False`):
 |              Whether to only load local model weights and configuration files or not. If set to `True`, the model
 |              won't be downloaded from the Hub.
 |          use_auth_token (`str` or *bool*, *optional*):
 |              The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
 |              `diffusers-cli login` (stored in `~/.huggingface`) is used.
 |          revision (`str`, *optional*, defaults to `"main"`):
 |              The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
 |              allowed by Git.
 |          subfolder (`str`, *optional*, defaults to `""`):
 |              The subfolder location of a model file within a larger model repository on the Hub or locally.
 |          return_unused_kwargs (`bool`, *optional*, defaults to `False):
 |              Whether unused keyword arguments of the config are returned.
 |          return_commit_hash (`bool`, *optional*, defaults to `False):
 |              Whether the `commit_hash` of the loaded configuration are returned.
 |      
 |      Returns:
 |          `dict`:
 |              A dictionary of all the parameters stored in a JSON configuration file.
 |  
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from diffusers.configuration_utils.ConfigMixin:
 |  
 |  config
 |      Returns the config of the class as a frozen dictionary
 |      
 |      Returns:
 |          `Dict[str, Any]`: Config of the class.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from diffusers.configuration_utils.ConfigMixin:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from diffusers.configuration_utils.ConfigMixin:
 |  
 |  has_compatibles = False
 |  
 |  ignore_for_config = []
 |  
 |  ----------------------------------------------------------------------
 |  Class methods inherited from diffusers.loaders.FromSingleFileMixin:
 |  
 |  from_ckpt(*args, **kwargs) from builtins.type
 |  
 |  from_single_file(pretrained_model_link_or_path, **kwargs) from builtins.type
 |      Instantiate a [`DiffusionPipeline`] from pretrained pipeline weights saved in the `.ckpt` or `.safetensors`
 |      format. The pipeline is set in evaluation mode (`model.eval()`) by default.
 |      
 |      Parameters:
 |          pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
 |              Can be either:
 |                  - A link to the `.ckpt` file (for example
 |                    `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
 |                  - A path to a *file* containing all pipeline weights.
 |          torch_dtype (`str` or `torch.dtype`, *optional*):
 |              Override the default `torch.dtype` and load the model with another dtype. If `"auto"` is passed, the
 |              dtype is automatically derived from the model's weights.
 |          force_download (`bool`, *optional*, defaults to `False`):
 |              Whether or not to force the (re-)download of the model weights and configuration files, overriding the
 |              cached versions if they exist.
 |          cache_dir (`Union[str, os.PathLike]`, *optional*):
 |              Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
 |              is not used.
 |          resume_download (`bool`, *optional*, defaults to `False`):
 |              Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
 |              incompletely downloaded files are deleted.
 |          proxies (`Dict[str, str]`, *optional*):
 |              A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
 |              'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
 |          local_files_only (`bool`, *optional*, defaults to `False`):
 |              Whether to only load local model weights and configuration files or not. If set to `True`, the model
 |              won't be downloaded from the Hub.
 |          use_auth_token (`str` or *bool*, *optional*):
 |              The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
 |              `diffusers-cli login` (stored in `~/.huggingface`) is used.
 |          revision (`str`, *optional*, defaults to `"main"`):
 |              The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
 |              allowed by Git.
 |          use_safetensors (`bool`, *optional*, defaults to `None`):
 |              If set to `None`, the safetensors weights are downloaded if they're available **and** if the
 |              safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
 |              weights. If set to `False`, safetensors weights are not loaded.
 |          extract_ema (`bool`, *optional*, defaults to `False`):
 |              Whether to extract the EMA weights or not. Pass `True` to extract the EMA weights which usually yield
 |              higher quality images for inference. Non-EMA weights are usually better for continuing finetuning.
 |          upcast_attention (`bool`, *optional*, defaults to `None`):
 |              Whether the attention computation should always be upcasted.
 |          image_size (`int`, *optional*, defaults to 512):
 |              The image size the model was trained on. Use 512 for all Stable Diffusion v1 models and the Stable
 |              Diffusion v2 base model. Use 768 for Stable Diffusion v2.
 |          prediction_type (`str`, *optional*):
 |              The prediction type the model was trained on. Use `'epsilon'` for all Stable Diffusion v1 models and
 |              the Stable Diffusion v2 base model. Use `'v_prediction'` for Stable Diffusion v2.
 |          num_in_channels (`int`, *optional*, defaults to `None`):
 |              The number of input channels. If `None`, it is automatically inferred.
 |          scheduler_type (`str`, *optional*, defaults to `"pndm"`):
 |              Type of scheduler to use. Should be one of `["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm",
 |              "ddim"]`.
 |          load_safety_checker (`bool`, *optional*, defaults to `True`):
 |              Whether to load the safety checker or not.
 |          text_encoder ([`~transformers.CLIPTextModel`], *optional*, defaults to `None`):
 |              An instance of `CLIPTextModel` to use, specifically the
 |              [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant. If this
 |              parameter is `None`, the function loads a new instance of `CLIPTextModel` by itself if needed.
 |          vae (`AutoencoderKL`, *optional*, defaults to `None`):
 |              Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations. If
 |              this parameter is `None`, the function will load a new instance of [CLIP] by itself, if needed.
 |          tokenizer ([`~transformers.CLIPTokenizer`], *optional*, defaults to `None`):
 |              An instance of `CLIPTokenizer` to use. If this parameter is `None`, the function loads a new instance
 |              of `CLIPTokenizer` by itself if needed.
 |          kwargs (remaining dictionary of keyword arguments, *optional*):
 |              Can be used to overwrite load and saveable variables (for example the pipeline components of the
 |              specific pipeline class). The overwritten components are directly passed to the pipelines `__init__`
 |              method. See example below for more information.
 |      
 |      Examples:
 |      
 |      ```py
 |      >>> from diffusers import StableDiffusionPipeline
 |      
 |      >>> # Download pipeline from huggingface.co and cache.
 |      >>> pipeline = StableDiffusionPipeline.from_single_file(
 |      ...     "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
 |      ... )
 |      
 |      >>> # Download pipeline from local file
 |      >>> # file is downloaded under ./v1-5-pruned-emaonly.ckpt
 |      >>> pipeline = StableDiffusionPipeline.from_single_file("./v1-5-pruned-emaonly")
 |      
 |      >>> # Enable float16 and move to GPU
 |      >>> pipeline = StableDiffusionPipeline.from_single_file(
 |      ...     "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt",
 |      ...     torch_dtype=torch.float16,
 |      ... )
 |      >>> pipeline.to("cuda")
 |      ```
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from diffusers.loaders.LoraLoaderMixin:
 |  
 |  unload_lora_weights(self)
 |      Unloads the LoRA parameters.
 |      
 |      Examples:
 |      
 |      ```python
 |      >>> # Assuming `pipeline` is already loaded with the LoRA parameters.
 |      >>> pipeline.unload_lora_weights()
 |      >>> ...
 |      ```
 |  
 |  write_lora_layers(state_dict: Dict[str, torch.Tensor], save_directory: str, is_main_process: bool, weight_name: str, save_function: Callable, safe_serialization: bool)
 |  
 |  ----------------------------------------------------------------------
 |  Class methods inherited from diffusers.loaders.LoraLoaderMixin:
 |  
 |  load_lora_into_text_encoder(state_dict, network_alphas, text_encoder, prefix=None, lora_scale=1.0) from builtins.type
 |      This will load the LoRA layers specified in `state_dict` into `text_encoder`
 |      
 |      Parameters:
 |          state_dict (`dict`):
 |              A standard state dict containing the lora layer parameters. The key should be prefixed with an
 |              additional `text_encoder` to distinguish between unet lora layers.
 |          network_alphas (`Dict[str, float]`):
 |              See `LoRALinearLayer` for more details.
 |          text_encoder (`CLIPTextModel`):
 |              The text encoder model to load the LoRA layers into.
 |          prefix (`str`):
 |              Expected prefix of the `text_encoder` in the `state_dict`.
 |          lora_scale (`float`):
 |              How much to scale the output of the lora linear layer before it is added with the output of the regular
 |              lora layer.
 |  
 |  load_lora_into_unet(state_dict, network_alphas, unet) from builtins.type
 |      This will load the LoRA layers specified in `state_dict` into `unet`.
 |      
 |      Parameters:
 |          state_dict (`dict`):
 |              A standard state dict containing the lora layer parameters. The keys can either be indexed directly
 |              into the unet or prefixed with an additional `unet` which can be used to distinguish between text
 |              encoder lora layers.
 |          network_alphas (`Dict[str, float]`):
 |              See `LoRALinearLayer` for more details.
 |          unet (`UNet2DConditionModel`):
 |              The UNet model to load the LoRA layers into.
 |  
 |  lora_state_dict(pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs) from builtins.type
 |      Return state dict for lora weights and the network alphas.
 |      
 |      <Tip warning={true}>
 |      
 |      We support loading A1111 formatted LoRA checkpoints in a limited capacity.
 |      
 |      This function is experimental and might change in the future.
 |      
 |      </Tip>
 |      
 |      Parameters:
 |          pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
 |              Can be either:
 |      
 |                  - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
 |                    the Hub.
 |                  - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
 |                    with [`ModelMixin.save_pretrained`].
 |                  - A [torch state
 |                    dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).
 |      
 |          cache_dir (`Union[str, os.PathLike]`, *optional*):
 |              Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
 |              is not used.
 |          force_download (`bool`, *optional*, defaults to `False`):
 |              Whether or not to force the (re-)download of the model weights and configuration files, overriding the
 |              cached versions if they exist.
 |          resume_download (`bool`, *optional*, defaults to `False`):
 |              Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
 |              incompletely downloaded files are deleted.
 |          proxies (`Dict[str, str]`, *optional*):
 |              A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
 |              'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
 |          local_files_only (`bool`, *optional*, defaults to `False`):
 |              Whether to only load local model weights and configuration files or not. If set to `True`, the model
 |              won't be downloaded from the Hub.
 |          use_auth_token (`str` or *bool*, *optional*):
 |              The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
 |              `diffusers-cli login` (stored in `~/.huggingface`) is used.
 |          revision (`str`, *optional*, defaults to `"main"`):
 |              The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
 |              allowed by Git.
 |          subfolder (`str`, *optional*, defaults to `""`):
 |              The subfolder location of a model file within a larger model repository on the Hub or locally.
 |          mirror (`str`, *optional*):
 |              Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
 |              guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
 |              information.
 |  
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from diffusers.loaders.LoraLoaderMixin:
 |  
 |  lora_scale
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from diffusers.loaders.LoraLoaderMixin:
 |  
 |  text_encoder_name = 'text_encoder'
 |  
 |  unet_name = 'unet'

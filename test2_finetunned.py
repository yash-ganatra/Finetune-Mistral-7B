
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Map user-friendly style keywords to full visual base styles
STYLE_MAP = {
    "cartoon": "in a cartoon-style educational environment with soft lighting, expressive characters, and clear candlestick illustrations",
    
    "pixel": "in a pixel-art visual style with pastel tones, bright daylight, and minimal arcade elements, showing clear, blocky candlestick charts",
    
    "realistic": "in a bright, modern trading workspace with natural daylight and clean architectural lines, showcasing realistic candlestick displays",
    
    "watercolor": "in a soft watercolor-painted landscape where candlestick charts blend naturally with gentle brushstrokes and light pastel tones",
    
    "comic": "in a comic-book panel with bold outlines, expressive characters, and white backgrounds for clarity, using red and green candlestick icons",
    
    "cyberpunk": "in a dark neon-lit cyberpunk trading bunker with glitchy data streams, glowing candlestick projections, and ambient tech aesthetics",
    
    "flat ui": "in a minimalist flat UI dashboard with clean chart grids, soft shadows, and subtle color-coded candlesticks on a bright background",
    
    "fantasy": "in a fantasy-inspired realm where glowing candlesticks float among magical scrolls and enchanted diagrams, surrounded by light auras",
    
    "infographic": "in a clean infographic layout with modular sections, labels, arrows, and candlestick illustrations in soft green and orange tones",

    "flat ui v2": "in a flat, cartoon-style vector illustration with thick outlines, soft colors, and minimalistic shapes designed for clear educational communication"
}

def generate_super_prompt(
    model_path: str,
    input_text: str,
    style_keyword: str = "realistic",  # e.g. "cartoon", "pixel", etc.
    colors: list = None,
    dark_mode: bool = False,
    strict_mode: bool = True,
    max_new_tokens: int = 200,
):
    """
    Generate a cinematic image prompt from a technical explanation using a fine-tuned Mistral model.

    Args:
        model_path (str): Path or HuggingFace model ID.
        input_text (str): The explanation to convert.
        style_keyword (str): User-friendly visual style keyword.
        colors (list): Optional dominant colors.
        dark_mode (bool): Whether the image should be dark-themed.
        strict_mode (bool): Whether to preserve exact meaning.
        max_new_tokens (int): Maximum generated tokens.

    Returns:
        str: Generated super prompt.
    """
    if colors is None:
        colors = []

    # === Tone Modifier ===
    if dark_mode:
        tone_modifier = "with deep shadows, cinematic contrast, and a high-tech ambient glow"
    else:
        tone_modifier = "with bright lighting, vibrant colors, and a lively, energetic atmosphere"

    # === Style Modifier ===
    base_style = STYLE_MAP.get(style_keyword.lower(), "with no specific visual style")
    color_text = f" using a color palette of {', '.join(colors)}" if colors else ""
    final_style = base_style + color_text

    # === Prompt Construction ===
    if strict_mode:
        system_prompt = f"""Convert the following explanation into a vivid, cinematic image description suitable for AI image generation.
Preserve the core meaning, tone, and essential details of the input. Render the image {final_style} and {tone_modifier}."""
        temperature = 0.7
        top_p = 0.85
    else:
        system_prompt = f"""Convert the following explanation into a vivid, cinematic image description suitable for AI image generation.
Render the image {final_style} and {tone_modifier}."""
        temperature = 0.9
        top_p = 0.95

    full_prompt = f"{system_prompt}\n\n{input_text}\n\nSuper Prompt:"

    # === Load model and tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    super_prompt = generated_text.split("Super Prompt:")[-1].strip()

    return super_prompt


# === EXAMPLE RUN ===
if __name__ == "__main__":
    model_path = "path/to/your/fine-tuned-mistral"

    result = generate_super_prompt(
        model_path='./mistral7b-finetuned',
        input_text="Decision-making often involves choosing between a safe, clear path and a risky, potentially dangerous one.",
        style_keyword="flat ui v2",
        colors=['yellow','green','brown'],
        dark_mode=False,
        strict_mode=True
    )

    print("\n🎨 Generated Super Prompt:\n")
    print(result)

import os, torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER = os.getenv("ADAPTER_MODEL", "akothari09/f1StrategyTrainer")

tokenizer = None
model = None

def load_model():
    """Load the model once on startup"""
    global tokenizer, model
    if model is not None:
        return

    try:
        print("Loading model... This may take a minute.")
        print(f"Base model: {BASE_MODEL}")
        print(f"Adapter: {ADAPTER}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        
        print("Loading base model...")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        print("Loading adapter...")
        model = PeftModel.from_pretrained(base, ADAPTER)
        model.eval()
        
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        raise e

def generate_strategy(driver, race, track_temp, air_temp, wind_speed, track_condition, max_tokens, temperature):
    """Generate F1 race strategy based on inputs"""
    
    try:
        # Load model if not already loaded
        load_model()
        
        # Create prompt from inputs
        prompt = f"""Generate an optimal Formula 1 race strategy for the following conditions:

Driver: {driver}
Race/Circuit: {race}
Track Temperature: {track_temp}°C
Air Temperature: {air_temp}°C
Wind Speed: {wind_speed} km/h
Track Condition: {track_condition}

Provide a detailed race strategy including tire choices, pit stop windows, and key considerations."""

        # Format with chat template
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                do_sample=temperature > 0,
                temperature=float(temperature) if temperature > 0 else 1.0,
                top_p=0.95,
                repetition_penalty=1.1,
            )

        # Decode output
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        
        # Try to extract just the assistant's response (remove the prompt)
        if "<|im_start|>assistant" in decoded:
            strategy = decoded.split("<|im_start|>assistant")[-1].strip()
            strategy = strategy.replace("<|im_end|>", "").strip()
        else:
            strategy = decoded
        
        return strategy
        
    except Exception as e:
        error_msg = f"Error generating strategy: {str(e)}\n\nPlease check the Space logs for details."
        print(f"ERROR in generate_strategy: {str(e)}")
        import traceback
        traceback.print_exc()
        return error_msg

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Base(), title="F1 Strategy AI") as demo:
    
    gr.Markdown("""
    # 🏎️ F1 Strategy AI
    ### AI-Powered Race Strategy Generator
    
    Enter race conditions below to generate an optimal Formula 1 race strategy using a fine-tuned Qwen2.5 model.
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Race Parameters")
            
            driver = gr.Textbox(
                label="Driver",
                placeholder="e.g., Max Verstappen",
                value="Max Verstappen"
            )
            
            race = gr.Textbox(
                label="Race / Circuit",
                placeholder="e.g., Monaco Grand Prix",
                value="Monaco Grand Prix"
            )
            
            with gr.Row():
                track_temp = gr.Number(
                    label="Track Temperature (°C)",
                    value=35,
                    minimum=0,
                    maximum=60
                )
                
                air_temp = gr.Number(
                    label="Air Temperature (°C)",
                    value=28,
                    minimum=0,
                    maximum=50
                )
            
            with gr.Row():
                wind_speed = gr.Number(
                    label="Wind Speed (km/h)",
                    value=15,
                    minimum=0,
                    maximum=100
                )
                
                track_condition = gr.Dropdown(
                    label="Track Condition",
                    choices=["dry", "damp"],
                    value="dry"
                )
            
            gr.Markdown("### Generation Settings")
            
            with gr.Row():
                max_tokens = gr.Slider(
                    label="Max New Tokens",
                    minimum=100,
                    maximum=800,
                    value=400,
                    step=50
                )
                
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1
                )
            
            generate_btn = gr.Button("🏁 Generate Strategy", variant="primary", size="lg")
            clear_btn = gr.Button("Clear", variant="secondary")
        
        with gr.Column():
            gr.Markdown("### Generated Strategy")
            
            output = gr.Textbox(
                label="Race Strategy",
                lines=25,
                placeholder="Your AI-generated race strategy will appear here...",
                show_copy_button=True
            )
    
    gr.Markdown("""
    ---
    **Note:** First generation may take 20-30 seconds as the model loads. Subsequent generations will be faster.
    
    **Model:** Qwen2.5-7B-Instruct fine-tuned on F1 race strategy data
    """)
    
    # Button actions
    generate_btn.click(
        fn=generate_strategy,
        inputs=[driver, race, track_temp, air_temp, wind_speed, track_condition, max_tokens, temperature],
        outputs=output
    )
    
    clear_btn.click(
        fn=lambda: ("", "", 35, 28, 15, "dry", 400, 0.7, ""),
        inputs=None,
        outputs=[driver, race, track_temp, air_temp, wind_speed, track_condition, max_tokens, temperature, output]
    )
    
    # Example inputs
    gr.Examples(
        examples=[
            ["Max Verstappen", "Monaco Grand Prix", 35, 28, 15, "dry", 400, 0.7],
            ["Lewis Hamilton", "Silverstone", 28, 22, 25, "dry", 400, 0.7],
            ["Charles Leclerc", "Monza", 40, 32, 10, "dry", 400, 0.7],
        ],
        inputs=[driver, race, track_temp, air_temp, wind_speed, track_condition, max_tokens, temperature],
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
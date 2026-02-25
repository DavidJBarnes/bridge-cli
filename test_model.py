#!/usr/bin/env python3
"""
Test script for the fine-tuned Spring Boot model
Run this after training to validate the model's responses
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

# Test prompts for Spring Boot coding
TEST_PROMPTS = [
    "Create a Spring Boot REST controller for Product management with GET, POST, PUT, DELETE endpoints",
    "Write a Spring Data JPA repository for Order entity with custom query methods",
    "Create a Spring Security configuration with JWT authentication",
    "Write a Spring Boot service class with transaction management for payment processing",
    "Create a Spring Boot global exception handler for REST APIs",
]


def load_model(base_model: str, lora_path: str = None):
    """Load the base model and optionally apply LoRA adapter."""
    print(f"Loading base model: {base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
    )
    
    if lora_path:
        print(f"Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_length: int = 2048) -> str:
    """Generate a response for the given prompt."""
    # Format as instruction
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response


def run_tests(model, tokenizer, prompts: list = None):
    """Run test prompts and display results."""
    prompts = prompts or TEST_PROMPTS
    
    print("\n" + "=" * 60)
    print("SPRING BOOT MODEL TESTING")
    print("=" * 60)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Test {i}/{len(prompts)} ---")
        print(f"Prompt: {prompt[:80]}...")
        print("-" * 40)
        
        response = generate_response(model, tokenizer, prompt)
        
        # Print first 500 chars of response
        print(f"Response:\n{response[:1000]}")
        if len(response) > 1000:
            print(f"... [{len(response) - 1000} more characters]")
        
        print("-" * 40)
    
    print("\n✅ Testing complete!")


def interactive_mode(model, tokenizer):
    """Run in interactive mode for custom prompts."""
    print("\n🚀 Interactive Mode - Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        prompt = input("\nEnter your Spring Boot coding request:\n> ")
        if prompt.lower() in ["quit", "exit", "q"]:
            break
        
        print("\nGenerating...")
        response = generate_response(model, tokenizer, prompt)
        print(f"\n{response}")


def main():
    parser = argparse.ArgumentParser(description="Test the fine-tuned Spring Boot model")
    parser.add_argument(
        "--base-model",
        default="deepseek-ai/deepseek-coder-6.7b-instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--lora-path",
        default="/workspace/outputs/spring-boot-coder",
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Test base model without LoRA (for comparison)",
    )
    args = parser.parse_args()
    
    lora_path = None if args.no_lora else args.lora_path
    model, tokenizer = load_model(args.base_model, lora_path)
    
    if args.interactive:
        interactive_mode(model, tokenizer)
    else:
        run_tests(model, tokenizer)


if __name__ == "__main__":
    main()

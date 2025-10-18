---
language:
- en
- bn
license: mit
tags:
- translation
- multilingual
- m2m100
- english-to-bengali
- opus100
- seq2seq
pipeline_tag: translation
widget:
- text: "Hello, how are you today?"
  example_title: "English to Bengali Translation"
- text: "I am studying artificial intelligence."
  example_title: "AI Study"
- text: "The weather is very nice today."
  example_title: "Weather"
---

# M2M100 English to Bengali Translation Model

A fine-tuned M2M100-418M model specifically optimized for English-to-Bengali translation using the OPUS100 dataset.

## Model Description

This model is based on Facebook's M2M100-418M multilingual translation model, fine-tuned on the OPUS100 Bengali-English parallel corpus. The model has been specifically optimized to improve translation quality from English to Bengali (বাংলা).

### Key Features

- **Base Model**: M2M100-418M (483.9M parameters)
- **Translation Direction**: English → Bengali
- **Training Dataset**: OPUS100 Bengali-English subset
- **Model Type**: Sequence-to-Sequence Transformer
- **Languages**: English (en), Bengali (bn)

## Performance

The model was evaluated using standard machine translation metrics:

| Metric | Pre-Fine-tuning | Post-Fine-tuning | Improvement |
|--------|----------------|------------------|-------------|
| **BLEU Score** | 3.05 | 2.91 | -4.77% |
| **METEOR Score** | 11.06 | 20.00 | +80.80% |

### Key Results

- **METEOR Score**: Significant improvement of 80.80% (from 11.06 to 20.00)
- **Translation Quality**: Enhanced semantic understanding and better word choice
- **Training**: 100 steps with learning rate 5e-5, batch size 8

## Usage

### Direct Usage

```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

# Load model and tokenizer
model_name = "Jayanta8509/m2m100-en-bn-finetuned-opus100"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def translate_en_to_bn(text):
    """Translate English text to Bengali"""
    tokenizer.src_lang = "en"
    tokenizer.tgt_lang = "bn"
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id("bn"),
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
    
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation

# Example usage
english_text = "Hello, how are you today?"
bengali_translation = translate_en_to_bn(english_text)
print(f"English: {english_text}")
print(f"Bengali: {bengali_translation}")
```

### Using Pipeline

```python
from transformers import pipeline

# Create translation pipeline
translator = pipeline("translation", 
                     model="Jayanta8509/m2m100-en-bn-finetuned-opus100",
                     tokenizer="Jayanta8509/m2m100-en-bn-finetuned-opus100")

# Translate text
result = translator("I am studying artificial intelligence.")
print(result)
```

## Example Translations

| English | Bengali |
|---------|---------|
| "Hello, how are you today?" | "হ্যালো, আপনি আজ কেমন?" |
| "I live in Bangladesh." | "আমি বাংলাদেশে বসবাস করি।" |
| "The weather is very nice today." | "বর্তমানে আবহাওয়া খুব ভালো।" |
| "This is a test translation." | "এটি একটি পরীক্ষা অনুবাদ।" |
| "How are you doing today?" | "তুমি আজ কি করো?" |

## Training Details

### Dataset
- **Source**: OPUS100 Bengali-English parallel corpus
- **Training Samples**: 2,000
- **Validation Samples**: 200
- **Test Samples**: 100
- **Format**: English → Bengali sentence pairs

### Training Configuration
- **Learning Rate**: 5e-5
- **Batch Size**: 8
- **Max Steps**: 100
- **Mixed Precision**: FP16 (when GPU available)
- **Optimizer**: AdamW
- **Weight Decay**: 0.01

### Hardware
- **GPU**: Tesla T4 (15.83 GB memory)
- **Framework**: PyTorch with Transformers
- **Training Time**: ~100 steps

## Model Architecture

The model uses the M2M100 architecture:
- **Encoder-Decoder**: Transformer-based
- **Parameters**: 483.9M
- **Attention**: Multi-head self-attention
- **Languages**: 100+ languages supported
- **Specialization**: Fine-tuned for English-Bengali

## Limitations and Bias

- The model may not perform well on very long sentences (>128 tokens)
- Performance may vary for domain-specific or technical content
- Training was done on a subset of OPUS100, which may limit generalization
- Some translations may contain repetitive patterns due to limited training steps

## Recommendations for Use

1. **Input Length**: Keep English sentences under 128 tokens for best results
2. **Domain**: Works best with general-purpose text; may need additional fine-tuning for specialized domains
3. **Post-processing**: Consider applying post-processing to clean up repetitive patterns
4. **Evaluation**: Always evaluate on your specific use case before deployment

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{m2m100-en-bn-finetuned-opus100,
  title={M2M100 English to Bengali Translation Model},
  author={Jayanta},
  year={2024},
  url={https://huggingface.co/Jayanta8509/m2m100-en-bn-finetuned-opus100}
}
```

## Acknowledgments

- Facebook AI Research for the original M2M100 model
- OPUS project for the parallel corpus
- Hugging Face for the Transformers library
- The open-source community for evaluation tools

## License

This model is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

**Model Repository**: [Jayanta8509/m2m100-en-bn-finetuned-opus100](https://huggingface.co/Jayanta8509/m2m100-en-bn-finetuned-opus100)

**Training Notebook**: Available in the repository for reproducibility and further experimentation.

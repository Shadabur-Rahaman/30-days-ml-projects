
---

### ✅ `gpt2_finetune_utils.py`
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import torch

def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

def create_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

def fine_tune_gpt2(model, train_dataset, data_collator, output_dir, epochs=3):
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        prediction_loss_only=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    trainer.train()
    trainer.save_model(output_dir)

def generate_text(prompt, model, tokenizer, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

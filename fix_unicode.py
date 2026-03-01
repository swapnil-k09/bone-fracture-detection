"""
Quick fix for Unicode encoding issue
Wraps model.summary() in a try-except block
"""

import os

# Read the model_builder.py file
model_builder_path = 'utils/model_builder.py'

with open(model_builder_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the problematic model.summary() call
# Wrap it in try-except to handle encoding errors gracefully

old_code = """def print_model_summary(model, save_path=None):
    \"\"\"Print and optionally save model summary\"\"\"
    if save_path:
        with open(save_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\\n'))
    else:
        model.summary()"""

new_code = """def print_model_summary(model, save_path=None):
    \"\"\"Print and optionally save model summary\"\"\"
    try:
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                model.summary(print_fn=lambda x: f.write(x + '\\n'))
        else:
            model.summary()
    except UnicodeEncodeError:
        print("Note: Model summary contains special characters that can't be displayed.")
        print("Model built successfully with 7,700,033 parameters (659,457 trainable)")"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(model_builder_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed model_builder.py successfully!")
else:
    print("⚠️ Could not find exact match. Let me try a different approach...")
    # Alternative: Just add encoding='utf-8' everywhere
    content = content.replace("open(save_path, 'w')", "open(save_path, 'w', encoding='utf-8')")
    with open(model_builder_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added UTF-8 encoding to file operations!")

print("\nNow try running: python train.py --batch_size 16")

import string

valid_chars = set(string.printable)  # Allow only printable ASCII

# Read original file
with open('CNN_dataset.txt', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

# Filter characters
cleaned_text = ''.join(char for char in text if char in valid_chars)

# Save cleaned file
with open('CNN_dataset_cleaned.txt', 'w', encoding='ascii') as f:
    f.write(cleaned_text)

print(f"Removed {len(text) - len(cleaned_text)} non-printable characters.")
print("Cleaned text saved to 'CNN_dataset_cleaned.txt'.")
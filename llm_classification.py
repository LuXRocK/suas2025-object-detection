import ollama

def classify_image(img_path):
    response = ollama.chat(
        model = "llava:7b",
        
from ollama import chat
from pydantic import BaseModel

class Object(BaseModel):
    name: str
    confidance: float


path = "cropped_images/1.png"
objects_path = "objects.json"

response = chat(
    model="llava:7b",
    format=Object.model_json_schema(),
    messages=[
        {"role": "user",
         "content": f"Classify the object in the image, choose a object from the provided list, you can only use one object from the list, you can't calssify the object as 'other'. Return as JSON",
         "images": [path],
         "objects": [objects_path]
         }
    ],
    options={'temperature' : 0},
)

image_classification = Object.model_validate_json(response.message.content)
print(image_classification)
from fastapi import FastAPI
from transformers import MarianTokenizer, MarianMTModel

app = FastAPI()

# Translation model
translation_model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)

class TranslationRequest:
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the translation service!"}

@app.post("/translate")
def translate_text(request: TranslationRequest):
    try:
        inputs = tokenizer.encode(request.text, return_tensors="pt")
        translation = translation_model.generate(inputs, max_length=100)[0]
        translated_text = tokenizer.decode(translation, skip_special_tokens=True)
        return {"translation": translated_text}
    except Exception as e:
        return {"error": "Translation failed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)


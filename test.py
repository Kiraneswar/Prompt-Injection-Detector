from transformers import pipeline

classifier = pipeline("text-classification", model="./prompt_injection_detector")

print("🔐 Prompt Injection Detector (type 'exit' to quit)\n")

while True:
    prompt = input("Enter a prompt: ")

    if prompt.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    result = classifier(prompt)[0]
    label = result['label']
    score = result['score']

    label_text = "Injection 🚨" if label == "LABEL_1" else "Safe ✅"

    print(f"Prediction: {label_text} (confidence: {score:.4f})")
    print("-" * 60)

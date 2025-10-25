from transformers import pipeline

# 1. Load your trained model
classifier = pipeline("text-classification", model="./prompt_injection_detector")

print("🔐 Prompt Injection Detector (type 'exit' to quit)\n")

# 2. Take input from user repeatedly
while True:
    prompt = input("Enter a prompt: ")

    # Stop if user types exit
    if prompt.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    # 3. Run prediction
    result = classifier(prompt)[0]
    label = result['label']
    score = result['score']

    # Map labels to human-readable form
    label_text = "Injection 🚨" if label == "LABEL_1" else "Safe ✅"

    print(f"Prediction: {label_text} (confidence: {score:.4f})")
    print("-" * 60)

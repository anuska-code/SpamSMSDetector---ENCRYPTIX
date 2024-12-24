import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from spam_detector import predict_spam

def test_message():
    while True:
        print("\n=== SMS Spam Detector ===")
        print("Enter a message to test (or 'quit' to exit):")
        message = input("> ")
        
        if message.lower() == 'quit':
            print("Goodbye!")
            break
        
        result = predict_spam(message)
        print("\nResult:", result)
        
        if result == "SPAM":
            print("⚠️ This message appears to be SPAM!")
            print("Common spam characteristics detected: promotional content, urgency, or suspicious offers.")
        else:
            print("✅ This message appears to be legitimate (HAM).")
            print("Message characteristics are consistent with normal communication.")
        
        print("\nWould you like to test another message? (yes/no)")
        if input("> ").lower() != 'yes':
            print("Goodbye!")
            break

if __name__ == "__main__":
    print("Welcome to the SMS Spam Detector!")
    print("You can enter any message to check if it's spam or legitimate.")
    print("Type 'quit' at any time to exit.")
    test_message()

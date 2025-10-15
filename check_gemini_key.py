import os
import google.generativeai as genai
from dotenv import load_dotenv

def check_key():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables")
        return False
        
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Try to generate a simple test response
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content("Hello!")
        
        print("API key is valid!")
        print(f"Test response: {response.text}")
        return True
        
    except Exception as e:
        print(f"Error validating API key: {e}")
        return False

if __name__ == "__main__":
    check_key()
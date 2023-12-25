import os
import openai
import PyPDF2
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI()
# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = [page.extract_text() for page in reader.pages if page.extract_text() is not None]
    return "\n".join(text)

# Function to chunk text into parts with a given size
def chunk_text(text, chunk_size):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to generate questions from a text chunk
def generate_questions(text_chunk):
    messages = [{"role": "user", "content": f"Generate questions for student to answer based on the following text. Also notice that this question is given to student before provide them this documents: \n\n{text_chunk}."}]
    # try:
    response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use GPT-4's model identifier here
            messages = messages,
            temperature=0.7,
            # max_tokens=500,
            # n=1,
            # stop=["\n"]
        )
    return response.choices[0].message.content
    # except openai.error.OpenAIError as e:
    #     print(f"Error from OpenAI: {e}")
    #     return ""

# Main function to handle the PDF reading and question generation
def process_pdf_and_generate_questions(pdf_path, chunk_size=8192):
    full_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(full_text, chunk_size)
    questions = []
    
    for chunk in chunks:
        print(f"Generating questions for chunk: {chunk[:30]}...")  # Show progress
        qs = generate_questions(chunk)
        if qs:
            questions.append(qs)
            
    return questions

# Example usage
if __name__ == "__main__":
    pdf_file_path = 'cs224n-2019-notes02-wordvecs2.pdf'  # Replace with your PDF file path
    all_questions = process_pdf_and_generate_questions(pdf_file_path)
    
    print("Generated Questions:")
    with open("cs224n-2019-notes02-wordvecs2.txt", "a") as f:
        for i, questions in enumerate(all_questions, 1):
            # print(questions)
            f.write(questions)
            f.write("\n")

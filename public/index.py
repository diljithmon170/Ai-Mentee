import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LMS:
    def __init__(self):
        # Load LLaMA 2-7B model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

        # Define subjects and difficulty levels
        self.subjects = {
            1: "Python",
            2: "Java",
            3: "DBMS",
            4: "AI",
            5: "ML"
        }
        self.difficulty_levels = {
            1: "Beginner",
            2: "Intermediate",
            3: "Advanced"
        }

    def select_subject(self):
        print("Please select a subject you would like to learn:")
        for key, value in self.subjects.items():
            print(f"{key}. {value}")
        subject_choice = int(input("Enter the number corresponding to your choice: "))
        return self.subjects.get(subject_choice, None)

    def select_difficulty(self):
        print("Now, please select your difficulty level:")
        for key, value in self.difficulty_levels.items():
            print(f"{key}. {value}")
        difficulty_choice = int(input("Enter the number corresponding to your choice: "))
        return self.difficulty_levels.get(difficulty_choice, None)

    def generate_dynamic_content(self, prompt):
        # Encode the prompt and generate a response using the LLaMA model
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=500, num_return_sequences=1)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text.strip()

    def generate_course_plan(self, subject, difficulty):
        prompt = f"Generate a 30-day learning course plan for {subject} at {difficulty} level."
        return self.generate_dynamic_content(prompt)

    def topic_details(self, topic):
        prompt = f"Provide detailed learning materials and resources for the topic: {topic}."
        return self.generate_dynamic_content(prompt)

    def display_course_plan(self, plan):
        print("\nGenerated Course Plan:")
        print(plan)

    def start(self):
        print("Welcome to the AI-based Learning Management System!")
        subject = self.select_subject()
        if not subject:
            print("Invalid subject selection.")
            return
        difficulty = self.select_difficulty()
        if not difficulty:
            print("Invalid difficulty selection.")
            return

        print(f"\nGenerating course plan for {subject} - {difficulty} level...")
        course_plan = self.generate_course_plan(subject, difficulty)
        self.display_course_plan(course_plan)

        # Simulating topic selection for dynamic content generation
        topic_choice = input("\nSelect a topic to learn more or type 'exit' to quit: ")
        if topic_choice.lower() == 'exit':
            return

        topic_details = self.topic_details(topic_choice)
        print("\nDetailed Topic Learning Materials:")
        print(topic_details)


if __name__ == "__main__":
    lms = LMS()
    lms.start()
import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.models import UnansweredQuestion


question = UnansweredQuestion(
    question="How does vLLM work?"
)

print(question)
print(question.question_id)


q1 = UnansweredQuestion(question="Question 1")
q2 = UnansweredQuestion(question="Question 2")

print(q1.question_id)
print(q2.question_id)
print(q1.question_id == q2.question_id)


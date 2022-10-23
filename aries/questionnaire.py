import json

print("You will now answer some questions about the treatment.  ")

question = ['Do you have a cough ? ',
             "Do you have any gastric problems ? ",
             "Do you have headaches ? ",
             "Do you have dizziness  ? ",
             "Do you have chills ? ",
             "Do you have breathing difficulties ? ",
             "Do you feel depressed ? ",
             "Do you feel a loss of taste ? ",
             "Do you feel a loss of smell ? "
             ]
reponse = []

for questions in question:
    reponses = input(f"{questions} ?")
    reponse.append(reponses)
    print('Thank you very much for your participation, we are very grateful. We will contact you in the next few days. See you soon!')

with open('answers.json', 'w', encoding='utf-8') as f:
    json.dump(reponse, f, ensure_ascii=False, indent=4)
    
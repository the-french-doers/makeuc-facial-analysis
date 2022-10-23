import json

print("Vous allez desormais reepondre a quelques questions ")

question = ['avez vous de la toux',
             "avez vous des pbs gastriques",
             "maux de tetes vertiges frissons ",
             "dificultées respiratoires",
             "depression",
             "perte de gouts et d'odorat"
             ]
reponse = []

for questions in question:
    reponses = input(f"{questions} ?")
    reponse.append(reponses)

with open('answers.json', 'w', encoding='utf-8') as f:
    json.dump(reponse, f, ensure_ascii=False, indent=4)
    
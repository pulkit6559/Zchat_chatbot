checkpoint = ".chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initialiser())
saver = tf.train.Saver()

saver.restore(session, checkpoint)

def convert_string2int(question, word2int):
    question = clean_text(questions)
    return [word2int.get(word, word2int['<OUT']) for word in questions.split()]

while(True):
    question = input("you: ")
    if(question == "Googbye"):
        break
    question = convert_string2int(question, questionswords2int)
    questions = question + [questionswords2int['<PAD>']]* (20-len(questions))
    question = question + 

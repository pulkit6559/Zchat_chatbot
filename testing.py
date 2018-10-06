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
    fake_batch = np.zeros((batch_size, 20))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_probability : 0.5})[0]
    answer = ""
    for i in np.argmax(predicted_answer, 1):
        if answersints2word[i] == 'i':
            token = "I"
        elif answersints2word[i] == '<EOS':
            token = '.'
        elif answersints2word[i] == '<OUT>':
            tokec = 'out'
        else:
            token = ' ' + answersints2word[i]
        answer += token
        if token == '.':
            break
        
    print('ChatBot: ' + answer)
    
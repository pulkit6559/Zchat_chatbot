checkpoint = ".chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initialiser())
saver = tf.train.Saver()

saver.restore(session, checkpoint)
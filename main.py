import tensorflow as tf
import numpy as np
import random
import time
import sys
from tqdm import tqdm
from model_net import ModelNetwork, embed_to_vocab, decode_embed
import uuid

def main():

    # Prefix to prompt the network in test mode
    TEST_PREFIX = 'The '
    # path to training text
    DATA_PATH = "data/trump_speeches.txt"

    # number of total batches
    NUM_TRAIN_BATCHES = 500000

    lstm_size = 256 #128
    num_layers = 2
    batch_size =  10 #128
    time_steps = 10 #50

    data_ = ""
    with open(DATA_PATH, 'r') as f:
        data_ += f.read()
    data_ = data_.lower()



    ## Convert to 1-hot coding
    vocab = list(set(data_))

    data = embed_to_vocab(data_, vocab)

    in_size = out_size = len(vocab)


    LEN_TEST_TEXT = 500 # Number of test characters of text to generate after training the network
    # default 0.003
    learning_rate_prop = 0.0003

    ## Initialize the network
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config)

    net = ModelNetwork(in_size = in_size,
                        lstm_size = lstm_size,
                        num_layers = num_layers,
                        out_size = out_size,
                        session = sess,
                        learning_rate = learning_rate_prop,
                        name = "char_rnn_network")

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.global_variables())

    last_time = time.time()

    batch = np.zeros((batch_size, time_steps, in_size))
    batch_y = np.zeros((batch_size, time_steps, in_size))

    possible_batch_ids = range(data.shape[0]-time_steps-1)

    for i in tqdm(range(NUM_TRAIN_BATCHES)):
        # Sample time_steps consecutive samples from the dataset text file
        batch_id = random.sample( possible_batch_ids, batch_size )
        for j in range(time_steps):
            ind1 = [k+j for k in batch_id]
            ind2 = [k+j+1 for k in batch_id]

            batch[:, j, :] = data[ind1, :]
            batch_y[:, j, :] = data[ind2, :]


        cst = net.train_batch(batch, batch_y)
        if (i%100) == 0:
            new_time = time.time()
            diff = new_time - last_time
            last_time = new_time

            print("batch: ",i,"   loss: ",cst,"   speed: ",(100.0/diff)," batches / s | ", learning_rate_prop)
    model_uuid = str(uuid.uuid1())
    saver.save(sess, "model/"+model_uuid+"/model.ckpt")
    print("Finished training model, model id: " + model_uuid)

    print("Starting prediction on: " + model_uuid)

    TEST_PREFIX = TEST_PREFIX.lower()
    for i in range(len(TEST_PREFIX)):
        out = net.run_step( embed_to_vocab(TEST_PREFIX[i], vocab) , i==0)

    gen_str = TEST_PREFIX
    for i in range(LEN_TEST_TEXT):
        element = np.random.choice( range(len(vocab)), p=out ) # Sample character from the network according to the generated output probabilities

        gen_str += vocab[element]

        out = net.run_step( embed_to_vocab(vocab[element], vocab) , False )

    print ('----------------Text----------------')
    print (gen_str)
    print ('----------------End----------------')
    text_file = open("data/output.txt", "w")
    text_file.write(gen_str)
    text_file.close()

if __name__ == '__main__':
    main()

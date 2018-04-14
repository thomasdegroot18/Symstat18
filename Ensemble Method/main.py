
from gensim.models import Word2Vec
from helper import parameter_parser
from gensim.models import KeyedVectors
import numpy as np
import tensorflow as tf
import csv
import time
from datetime import datetime
import random
import pickle

import Embedding_models.Diff2Vec.diffusion_2_vec as diffmain 
import Embedding_models.Node2Vec.main as nodemain 
import Embedding_models.Struc2Vec.main as strucmain
import Embedding_models.Infuser2Vec.main as infusermain
from Neural_model import LinkPredictionModel


def ensemble_method_StrucDiff2vec():
    # Run both embeddings.
    no_ensemble_method_struc2vec()
    no_ensemble_method_diff2vec()
    return "Ensemble method finished Finished"


def ensemble_infuser_method_StrucDiff2vec():
    infusermain.main(args)
    return "Infuser Finished"


def no_ensemble_method_struc2vec():
    strucmain.main(args)
    return


def no_ensemble_method_diff2vec():
    diffmain.main(args)
    return


def no_ensemble_method_node2vec():
    nodemain.main(args)
    return

def hadamard_transform(subject, object_):

    return np.multiply(subject, object_)

def dataset_preprocess_hadamard(dataset_pos, dataset_neg):
    # If not using the negative set the values are also no needed. 
    if dataset_neg == None:
        dataset_pos = hadamard_transform(dataset_pos[0], dataset_pos[2])
        targets = np.ones([dataset_pos.shape[0], 1])
        return dataset_pos, targets

    dataset_pos = hadamard_transform(dataset_pos[0], dataset_pos[2])
    dataset_neg = hadamard_transform(dataset_neg[0], dataset_neg[2])
    dataset = np.empty([2*min(dataset_pos.shape[0], dataset_neg.shape[0]), dataset_neg.shape[1]])
    targets = np.empty([2*min(dataset_pos.shape[0], dataset_neg.shape[0]), 2])
    i_neg = 0
    i_pos = 0
    dataset_size = min(dataset_pos.shape[0], dataset_neg.shape[0])
    for row_range in range(0,2*dataset_size):
        if i_neg < dataset_size and i_pos < dataset_size:
            if random.random() < 0.5:
                # Positive sample
                dataset[row_range] = dataset_pos[i_pos]
                targets[row_range] = [1, 0]
                i_pos += 1

            else:
                # Negative sample
                dataset[row_range] = dataset_neg[i_neg]
                targets[row_range] = [0, 1]
                i_neg += 1

        elif i_neg >= dataset_size:
            # IF a set is exhuasted the last are filled up with the other datset
            dataset[row_range:2*dataset_size] = dataset_pos[i_pos:dataset_size]
            targets[row_range:2*dataset_size] = np.concatenate([np.ones([2*dataset_size-row_range,1]),np.zeros([2*dataset_size-row_range,1])],axis=1)

            break
        elif i_pos >= dataset_size:
            dataset[row_range:2*dataset_size] = dataset_neg[i_neg:dataset_size]
            targets[row_range:2*dataset_size] = np.concatenate([np.zeros([2*dataset_size-row_range,1]),np.ones([2*dataset_size-row_range,1])],axis=1)

            break


    return dataset, targets


def dataset_transformer(embeddings_model, datasetinput, inputsize, usenegativesample, buidnegativesample):
    # Build array to store the embeddings of all the dataelements
    with open(datasetinput, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        row_count = sum(1 for row in reader) 

    subject = np.empty([row_count, inputsize])
    relation = np.empty([row_count, inputsize])
    Object = np.empty([row_count, inputsize])

    incorrect_subject = True
    valid_triples = {}
    # Check if model uses strucdiff2vec or different things
    if isinstance(embeddings_model, list):
        # Loop through datset
        with open(datasetinput, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            i = 0
            for row in reader:
                try:
                    emb_1 = embeddings_model[0][str(row[0])]
                    emb_2 = embeddings_model[1][str(row[0])]
                    # Try to find word in embedding. 
                    subject[i] = np.concatenate((emb_1, emb_2), axis=0)
                except:
                    print("Subject Failed")
                    incorrect_subject = False
                    subject[i] = np.zeros([inputsize])
                try:
                    # Try to find word in embedding. 
                    emb_1 = embeddings_model[0][str(row[1])]
                    emb_2 = embeddings_model[1][str(row[1])]
                    relation[i] = np.concatenate((emb_1, emb_2), axis=0)
                except:
                    relation[i] = np.zeros([inputsize])
                try:
                    # Try to find word in embedding. 
                    emb_1 = embeddings_model[0][str(row[2])]
                    emb_2 = embeddings_model[1][str(row[2])]
                    if incorrect_subject:
                        if str(row[0]) in valid_triples.keys():
                            valid_triples[str(row[0])].add(str(row[2]))
                        else:
                            valid_triples[str(row[0])] = set(str(row[2]))
                    Object[i] = np.concatenate((emb_1, emb_2), axis=0)
                except:
                    print("Object Failed")
                    Object[i] = np.zeros([inputsize])
                i += 1
                incorrect_subject = True

    else:
        with open(datasetinput, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            i = 0
            for row in reader:
                try:
                    # Try to find word in embedding. 
                    subject[i] = embeddings_model[str(row[0])]
                except:
                    print("Subject Failed")
                    incorrect_subject = False
                    subject[i] = np.zeros([inputsize])
                try:
                    # Try to find word in embedding. 
                    relation[i] = embeddings_model[str(row[1])]
                except:
                    relation[i] = np.zeros([inputsize])
                try:
                    # Try to find word in embedding. 
                    Object[i] = embeddings_model[str(row[2])]
                    if incorrect_subject:
                        if str(row[0]) in valid_triples.keys():
                            valid_triples[str(row[0])].add(str(row[2]))
                        else:
                            valid_triples[str(row[0])] = set(str(row[2]))
                except:
                    print("Object Failed")
                    Object[i] = np.zeros([inputsize])
                i += 1
                incorrect_subject = True


    embedding_links = [subject, relation, Object]

    ## If not useing negative sampling
    if not(usenegativesample):
        return embedding_links, None

    # Building a storage for negative link set, such that all neural networks are trained on same dataset. For continuity
    no_links = []

    # Building a negative linksdataset, by checking the unique valid triples and making sure the right links are not connected
    neg_sample_size = len(valid_triples.keys()) * int(row_count/len(valid_triples.keys()))
    subject_neg = np.empty([neg_sample_size, inputsize])
    relation_neg = np.empty([neg_sample_size, inputsize])
    Object_neg = np.empty([neg_sample_size, inputsize])

    
    if buidnegativesample:

        i = 0

        for neg_size in range(0, int(row_count/len(valid_triples.keys()))):
            for key in valid_triples.keys():
                if isinstance(embeddings_model, list):
                    emb_1 = embeddings_model[0][key]
                    emb_2 = embeddings_model[1][key]
                    subject_neg[i] = np.concatenate((emb_1, emb_2), axis=0)
                else:
                    subject_neg[i] = embeddings_model[key]
                # No different links, so can be added in later.
                relation_neg[i] = np.zeros([inputsize])
                # Find a good object
                object_sampled = False
                while not(object_sampled):
                    sampled_object = list(valid_triples.keys())[random.randint(0,len(valid_triples.keys())-1)]
                    if not(sampled_object in valid_triples[key]) and not(key in valid_triples[sampled_object]):
                        if isinstance(embeddings_model, list):
                            emb_1 = embeddings_model[0][sampled_object]
                            emb_2 = embeddings_model[1][sampled_object]
                            Object_neg[i] = np.concatenate((emb_1, emb_2), axis=0)

                        else:
                            Object_neg[i] = embeddings_model[sampled_object]
                        object_sampled = True
                        no_links.append([key,"No-relation",sampled_object])

                i += 1


        with open(datasetinput[:-4] + '_neg.pickle', 'wb') as handle:
            pickle.dump(no_links, handle, protocol=3)
    else:
        ## Load the pickled dataset.
        with open(datasetinput[:-4] + '_neg.pickle', 'rb') as handle:
            no_links = pickle.load(handle)
            i = 0
            for no_link_elem in no_links:
                if isinstance(embeddings_model, list):
                    emb_1 = embeddings_model[0][no_link_elem[0]]
                    emb_2 = embeddings_model[1][no_link_elem[0]]
                    subject_neg[i] = np.concatenate((emb_1, emb_2), axis=0)
                    emb_1 = embeddings_model[0][no_link_elem[2]]
                    emb_2 = embeddings_model[1][no_link_elem[2]]
                    Object_neg[i] = np.concatenate((emb_1, emb_2), axis=0)
                    relation_neg[i] = np.zeros([inputsize])
                else:
                    subject_neg[i] = embeddings_model[no_link_elem[0]]
                    # No different links, so can be added in later.
                    relation_neg[i] = np.zeros([inputsize])
                    Object_neg[i] = embeddings_model[no_link_elem[2]]
                i+= 1

    embedding_no_links = [subject_neg, relation_neg, Object_neg]

    return embedding_links, embedding_no_links



def train(dataset, testset, validset, args):

    args.outputsize = 2
    args.batch_size = 64
    # Initialize the model
    model = LinkPredictionModel(
        batch_size = args.batch_size,
        prediction_pair = args.dimensions,
        num_hidden = args.dimensionsHidden,
        num_layers = args.Hidden,
        output_size = args.outputsize
    )

    # Initialize the global step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)
    starter_learning_rate = args.learning_rate

    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               50000, 0.96, staircase=True)
    # Generate the model
    logits = model._run_model(model.inputs)
    loss = model._compute_loss(model.targets)

    accuracy = model._compute_accuracy(model.targets)
    # Define the optimizer
    optimizer = tf.train.RMSPropOptimizer(learning_rate)

    # Compute the gradients for each variable
    grads_and_vars = optimizer.compute_gradients(model.loss)
    # train_op = optimizer.apply_gradients(grads_and_vars, global_step)
    grads, variables = zip(*grads_and_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=args.max_norm_gradient)
    apply_gradients_op = optimizer.apply_gradients(zip(grads_clipped, variables), global_step=global_step)
    
    # Generate the session
    sess = tf.Session()

    # Initialize the global variables by running the session once
    sess.run(tf.global_variables_initializer())

    trainingset_size = int(dataset[0].shape[0]/args.batch_size)-1
    testset_size = int(testset[0].shape[0]/args.batch_size)-1

    for train_step in range(int(args.train_steps)):

        train_batch = random.randint(1,trainingset_size)
        # # Learn the batches
        # Run a trainingstep once.

        sess.run(apply_gradients_op, feed_dict={model.inputs: dataset[0][(train_batch-1)*args.batch_size:train_batch*args.batch_size],
                                                model.targets: dataset[1][(train_batch-1)*args.batch_size:train_batch*args.batch_size]
                                                })




        # Output the training progress
        if train_step % args.print_every == 0:

            test_batch = random.randint(1,testset_size)

            # Test batches
            # Return loss and accuracy

            _loss, _accuracy = sess.run([model.loss, model.accuracy], feed_dict={model.inputs: testset[0][(test_batch-1)*args.batch_size:test_batch*args.batch_size],
                                                                                model.targets: testset[1][(test_batch-1)*args.batch_size:test_batch*args.batch_size]
                                                                                })


            print("[{}] Train Step {:04d}/{:04d},  Loss =  {:.4f},  accuracy =  {:.4f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), train_step+1,
                int(args.train_steps), _loss, _accuracy
            ))

    # Validation
    total_accuracy = []

    for i in range(1,int(validset[0].shape[0]/args.batch_size)-1):
        _accuracy =sess.run(model.accuracy, feed_dict={model.inputs: validset[0][(i-1)*args.batch_size:i*args.batch_size],
                                                                                                model.targets: validset[1][(i-1)*args.batch_size:i*args.batch_size]
                                                                                                })
        total_accuracy.append(_accuracy)
    print("accuracy of the validationset is =  {:.4f}".format(sum(total_accuracy)/len(total_accuracy)))

    return sum(total_accuracy)/len(total_accuracy)


def main(args):

    # Build the correct ensembly information

    print("Building Embeddings")

    if args.ensembler == "Infuser":
        if not(args.skipensembler):
            ensemble_infuser_method_StrucDiff2vec()
        output = args.output + "Infuser" + str(args.dimensions) +".emb"
        ## Decide ensemble method to choose between multiple vocabs or infuser
        embeddings_model = KeyedVectors.load_word2vec_format(output)

    elif args.ensembler == "strucdiff2vec":
        if not(args.skipensembler):
            ensemble_method_StrucDiff2vec()
        args.dimensions = int(args.dimensions/2)
        output1 = args.output + "struc2vec" + str(args.dimensions) +".emb"
        ## Decide ensemble method to choose between multiple vocabs or infuser
        embeddings_model_struc = KeyedVectors.load_word2vec_format(output1)

        output2 = args.output + "diff2vec" + str(args.dimensions) +".emb"
        ## Decide ensemble method to choose between multiple vocabs or infuser
        embeddings_model_diff = KeyedVectors.load_word2vec_format(output2)

        embeddings_model = [embeddings_model_struc, embeddings_model_diff]
        args.dimensions = int(args.dimensions*2)

    elif args.ensembler == "Skip ensembler":
        output = "Embeddings/file.emb"
        ## Decide ensemble method to choose between multiple vocabs or infuser
        embeddings_model = KeyedVectors.load_word2vec_format(output)

    else: 
        if args.ensembler == "node2vec":
            if not(args.skipensembler):
                no_ensemble_method_node2vec()
            output = args.output + "node2vec" + str(args.dimensions) +".emb"
            ## Decide ensemble method to choose between multiple vocabs or infuser
            embeddings_model = KeyedVectors.load_word2vec_format(output)

        elif args.ensembler == "struc2vec":
            if not(args.skipensembler):
                no_ensemble_method_struc2vec()
            output = args.output + "struc2vec" + str(args.dimensions) +".emb"
            ## Decide ensemble method to choose between multiple vocabs or infuser
            embeddings_model = KeyedVectors.load_word2vec_format(output)

        elif args.ensembler == "diff2vec":
            if not(args.skipensembler):
                no_ensemble_method_diff2vec()
            output = args.output + "diff2vec" + str(args.dimensions) +".emb"
            ## Decide ensemble method to choose between multiple vocabs or infuser
            embeddings_model = KeyedVectors.load_word2vec_format(output)
        else:
            print("No ensembly method chosen try again")
            return 


    print("Loaded Embeddings")
    ## Transform the dataset to be trained in the model.
    dataset_pos, dataset_neg = dataset_transformer(embeddings_model, args.datasetinput, args.dimensions, args.usenegativesample, args.buildnegativesample)
    ## Transform the testset to be trained in the model.
    testset_pos, testset_neg = dataset_transformer(embeddings_model, args.testsetinput, args.dimensions, args.usenegativesample, args.buildnegativesample)
    ## Transform the validset to be trained in the model.
    validset_pos, validset_neg = dataset_transformer(embeddings_model, args.validsetinput, args.dimensions, args.usenegativesample, args.buildnegativesample)

    ## Preprocess data
    dataset, targets = dataset_preprocess_hadamard(dataset_pos, dataset_neg)
    testset, testtargets = dataset_preprocess_hadamard(testset_pos, testset_neg)
    validset, validtargets = dataset_preprocess_hadamard(validset_pos, validset_neg)

    print("Building model")
    ## build mode with tensorflow

    print("Training Model")
    ## Run the training on the dataset.

    model_accuracy = train([dataset, targets],[testset, testtargets],[validset, validtargets], args)
    return model_accuracy




if __name__ == "__main__":
    args = parameter_parser()

    # Amount of loops through embeddingstyle.
    average = 1

    # fb15k
    possible_embedding = [["strucdiff2vec", 128], ["strucdiff2vec", 256], ["Infuser",128], ["node2vec",128], ["struc2vec", 64], ["struc2vec", 128], ["diff2vec", 64], ["diff2vec", 128]]
    args.datasetinput = "datasets/fb15k/train.csv"
    args.testsetinput = "datasets/fb15k/test.csv"
    args.validsetinput = "datasets/fb15k/valid.csv"
    args.input = "datasets/fb15k/train.csv"
    args.output = "Embeddings/embedding_"
    args.walkloc = "random_walks_fb15k.txt"

    accuracy = {}

    for elem in possible_embedding:
        args.ensembler = elem[0]
        args.dimensions = elem[1]
        key = (elem[0],elem[1],"fb15k")
        accuracy_iter = []
        for iteration in range(0,average):

            accuracy_iter.append(main(args))
            tf.reset_default_graph()
        accuracy[key] = accuracy_iter

    # wn18
    args.datasetinput = "datasets/wn18/train.csv"
    args.testsetinput = "datasets/wn18/test.csv"
    args.validsetinput = "datasets/wn18/valid.csv"
    args.input = "datasets/wn18/train.csv"
    args.output = "Embeddings/embedding_wordnet_"
    args.walkloc = "random_walks_wn18.txt"    
    for elem in possible_embedding:
        args.ensembler = elem[0]
        args.dimensions = elem[1]
        key = (elem[0],elem[1],"wn18")

        accuracy_iter = []
        for iteration in range(0,average):

            accuracy_iter.append(main(args))
            tf.reset_default_graph()
        accuracy[key] = accuracy_iter

    for key in accuracy.keys():
        print("average accuracy of model "+key[0]+" with Dimensions: "+str(key[1])+" On dataset: "+ key[2]+ " given "+ str(average) + " iterations is =  {:.4f}".format(sum(accuracy[key])/len(accuracy[key])))

    for key in accuracy.keys():
        print(key, accuracy[key])

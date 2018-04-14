import argparse

def parameter_parser():

    """
    A method to parse up command line parameters. By default it gives an embedding of the Facebook Restaurants network.
    The default hyperparameters give a high quality representation already without grid search.
    """
    parser = argparse.ArgumentParser(description = "Run emsembler2vec.")

############# NEURAL NETWORK ARGUMENTS #########################

    parser.add_argument('--dimensionsHidden', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--Hidden', type=int, default=2,
                        help='Number of layer. Default is 1.')

    parser.add_argument('--datasetinput', nargs='?', default='datasets/fb15k/train.csv',
                        help='Input training dataset path')

    parser.add_argument('--testsetinput', nargs='?', default='datasets/fb15k/test.csv',
                        help='Input training testset path')

    parser.add_argument('--validsetinput', nargs='?', default='datasets/fb15k/valid.csv',
                        help='Input training validset path')

    parser.add_argument('--NNoutput', nargs='?', default=' ',
                        help='Input Embeddings path')

    parser.add_argument('--learning_rate', type=float, default=0.025, help='Learning rate')

    parser.add_argument('--train_steps', type=int, default=500000, help='Number of training steps')

    parser.add_argument('--max_norm_gradient', type=float, default=10.0, help='--')

    parser.add_argument('--print_every', type=int, default=1000, help='--')

    parser.add_argument('--batch_size', type=int, default=64, help='--')

    parser.add_argument('--usenegativesample', nargs='?', default=True)    # use negative sample

    parser.add_argument('--buildnegativesample', nargs='?', default=False)    # Building negative sample
############# EMBEDDINGS ARGUMENTS #########################

    parser.add_argument('--skipensembler', nargs='?', default=True,    # Skip ensembler
                        help='Skip ensembler')

    parser.add_argument('--ensembler', nargs='?', default='Infuser',    # Skip ensembler, strucdiff2vec, Infuser, node2vec, struc2vec, diff2vec
                        help='Input Embeddings path')

    parser.add_argument('--input', nargs='?', default='datasets/fb15k/train.csv', # fb15k, wn18
                        help='Input Embeddings path')

    parser.add_argument('--output', nargs='?', default='Embeddings/embedding_', # embedding_wordnet_
                        help='output Embeddings path')

    parser.add_argument('--walkloc', nargs='?', default='random_walks_fb15k.txt', # random_walks_wn18.txt, random_walks_fb15k.txt
                        help='output Embeddings path')

    parser.add_argument('--dimensions',
                        type = int,
                        default = 128,
                    help = 'Number of dimensions. Default is 128.')

    parser.add_argument('--vertex-set-cardinality',
                        type = int,
                        default = 40,
                    help = 'Length of diffusion per source is 2*cardianlity-1. Default is 40.')

    parser.add_argument('--num-diffusions',
                        type = int,
                        default = 10,
                    help = 'Number of diffusions per source. Default is 10.')

    parser.add_argument('--window-size',
                        type = int,
                        default = 10,
                        help = 'Context size for optimization. Default is 10.')

    parser.add_argument('--iter',
                        default = 5,
                        type = int,
                        help = 'Number of epochs in ASGD. Default is 1.')

    parser.add_argument('--workers',
                        type = int,
                        default = 8,
                    help = 'Number of cores. Default is 4.')

    parser.add_argument('--alpha',
                        type = float,
                        default = 0.025,
                    help = 'Initial learning rate. Default is 0.025.')
    
    parser.add_argument('--type',
                        type = str,
                        default = "eulerian",
                        help = 'Traceback type. Default is Eulerian.')


    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')


    parser.add_argument('--until-layer', type=int, default=None,
                        help='Calculation until the layer.')


    parser.add_argument('--weighted', dest='weighted', default=True, action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')

    parser.add_argument('--unweighted', dest='unweighted', action='store_false')

    parser.add_argument('--directed', dest='directed', default=False, action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')

    parser.add_argument('--OPT1', default=True, type=bool,
                      help='optimization 1')
    parser.add_argument('--OPT2', default=True, type=bool,
                      help='optimization 2')
    parser.add_argument('--OPT3', default=False, type=bool,
                      help='optimization 3')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')





    return parser.parse_args()

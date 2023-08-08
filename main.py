import argparse
# Create the parser
import logging


from utils import *
import gc
import torch
from train import *
import math


OPTIMIZERS = ['adam', 'sgd']
SPLITS = ['iid','non_iid']
def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',
                        help='name of optimizer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='adam')

    parser.add_argument('--data_split',
                        help='Data Split type;',
                        type=str,
                        choices=SPLITS,
                        default='iid')

    parser.add_argument('--num_rounds',
                        help='number of communication rounds to simulate;',
                        type=int,
                        default=400)

    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=4)
    parser.add_argument('--num_epochs', 
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=5)

    parser.add_argument('--learning_rate',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.003)

    parser.add_argument('--poison',
                        help='Poison level in Percentage',
                        type=int,
                        default=20)
    


    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))



    # load selected model
    print( style.YELLOW+" \n OPTIMIZER:"+ str(parsed["optimizer"])+" \n Data Split:"+ str(parsed["data_split"])+"\n COMM_ROUNDS:"+ str(parsed["num_rounds"])+" \n POISON CLIENTS:"+ str(parsed["clients_per_round"])+" \n BATCH SIZE:"+ str(parsed["batch_size"])+" \n LOCAL EPOCHS:"+ str(parsed["num_epochs"])+" \n LEARNING RATE:"+ str(parsed["learning_rate"])+" \n POISON LEVEL:"+ str(parsed["poison"]))


    return parsed


def main():
    # parser = argparse.ArgumentParser()
    # Add an argument
    # parser.add_argument('--name', type=str, required=True)
    # Parse the argument
    
    parsed =  read_options()
    
    poison_level = parsed["poison"]
    num_clients= parsed["clients_per_round"]
    poison_clients =  math.ceil( num_clients * (poison_level/100))
    print(poison_clients)
    if(poison_clients> num_clients ):
        logging.error("Poison level can not be greater than 100 %")
    else:
        train(batch_size= parsed["batch_size"]
     ,poison=poison_clients,data_split=parsed["data_split"],optimizer=parsed["optimizer"],comm_rounds = parsed["num_rounds"], local_epochs= parsed["num_epochs"], lr= parsed["learning_rate"], num_clients= parsed["clients_per_round"]  )




if __name__ == '__main__':
  
    gc.collect()
    torch.cuda.empty_cache()
    main()
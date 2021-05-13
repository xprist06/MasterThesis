# -----------------------------------------------------------------------------
# This software was developed as practical part of Master's thesis at FIT BUT
# The program uses multiobjective NSGA-II algorithm for designing accurate
# and compact CNNs.
#
# Author: Jan Pristas, xprist06@stud.fit.vutbr.cz
# Institute: Faculty of Information Technology, Brno University of Technology
#
# File: main.py
# Description: Process input arguments from command line, set evolution params
# and start computation
# -----------------------------------------------------------------------------


from evolution_search.nsga_ii import NSGAII
from evolution_search.result_export import ResultExport

import sys
import getopt
import logging


def help_msg():
    """
    Create help message for command line
    :return: help message
    """
    msg = "\nUsage: \n"
    msg += "\tmain.py\n"
    msg += "\tmain.py -h | --help\n"
    msg += "\tmain.py [options]\n"
    msg += "Options:\n"
    msg += "\t-h --help\tShow help message.\n"
    msg += "\t-p --pop \tNumber of individuals in each generation (integer > 1) [default: 15].\n"
    msg += "\t-g --gen \tNumber of generations (integer > 0) [default: 15].\n"
    msg += "\t-m --mut \tMutation probability (float 0-1) [default: 0.15].\n"
    msg += "\t--phases \tNumber of phases in genotypes (integer > 1) [default: 2].\n"
    msg += "\t--modules\tNumber of modules in genotypes (integer 1-10) [default: 6].\n"
    return msg


def arg_error(msg):
    """
    Print out error and help message
    :param msg: error message
    """
    logging.error(msg)
    logging.error(help_msg())
    sys.exit(1)


def main(argv):
    """
    Process input arguments from command line, set evolution params and start computation
    :param argv: command line arguments
    """
    population = 15
    generations = 15
    mutation = 0.15
    phases_cnt = 2
    modules_cnt = 6

    logging.basicConfig(level=logging.INFO)

    # Get input arguments
    try:
        opts, args = getopt.getopt(argv, "hp:g:m:", ["help", "pop=", "gen=", "mut=", "phases=", "modules="])
    except getopt.GetoptError:
        logging.error(help_msg())
        sys.exit(1)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            logging.info(help_msg())
            sys.exit()
        elif opt in ("-p", "--pop"):
            try:
                population = int(arg)
                if population < 2:
                    arg_error("Population value has to be an integer greater than 1. Value given: " + arg)
            except ValueError:
                arg_error("Population value has to be an integer. Value given: " + arg)
        elif opt in ("-g", "--gen"):
            try:
                generations = int(arg)
                if generations < 1:
                    arg_error("Generations value has to be an integer grater than 0. Value given: " + arg)
            except ValueError:
                arg_error("Generations value has to be an integer. Value given: " + arg)
        elif opt in ("-m", "--mut"):
            try:
                mutation = float(arg)
                if mutation < 0 or mutation > 1:
                    arg_error("Mutation probability value has to be a float between 0 and 1. Value given: " + arg)
            except ValueError:
                arg_error("Mutation probability value has to be a float. Value given: " + arg)
        elif opt == "--phases":
            try:
                phases_cnt = int(arg)
                if phases_cnt < 1:
                    arg_error("Phases count value has to be an integer greater than 0. Value given: " + arg)
            except ValueError:
                arg_error("Phases count value has to be an integer. Value given: " + arg)
        elif opt == "--modules":
            try:
                modules_cnt = int(arg)
                if modules_cnt < 1 or modules_cnt > 10:
                    arg_error("Modules count value has to be an integer between 1 and 10. Value given: " + arg)
            except ValueError:
                arg_error("Modules count value has to be an integer. Value given: " + arg)

    # Evolution parameters
    pop_size = population            # Number of individuals in population
    generations = generations        # Number of generations
    tournament_count = 0             # Type of selection, if 0, roulette is used
    mutation_probability = mutation  # Mutation probability
    phases = phases_cnt              # Number of phases in genotype
    modules = modules_cnt            # Number of modules in genotype
    genes_cros = True                # Genes crossover; if False, Modules crossover is used
    dataset = 2                      # 0 - MNIST, 1 - FASHION MNIST, 2 - SVHN, 3 - CIFAR10, 4 - CIFAR100
    batch_size = 128                 # Batch size for evolution
    epochs = 10                      # Number of epochs for evolution
    val_batch_size = 64              # Batch size for validation
    val_epochs = 50                  # Number of epochs for validation
    val_split = 0.2                  # Validation split of training data
    verbose = 2                      # Verbose

    # Types of information for export
    result_export = ResultExport(
        pareto_graph=True,
        keras_graph=True,
        keras_model=True,
        genotype_info=True
    )

    NSGAII(
        pop_size=pop_size,
        max_gen=generations,
        tournament_count=tournament_count,
        mutation_probability=mutation_probability,
        phase_count=phases,
        modules_count=modules,
        genes_cros=genes_cros,
        dataset=dataset,
        batch_size=batch_size,
        epochs=epochs,
        val_batch_size=val_batch_size,
        val_epochs=val_epochs,
        val_split=val_split,
        verbose=verbose,
        result_export=result_export
    )


if __name__ == "__main__":
    main(sys.argv[1:])

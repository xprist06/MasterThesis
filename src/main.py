from evolution_search.nsga_ii import NSGAII
from evolution_search.result_export import ResultExport

import sys
import getopt
import logging


def main(argv):
    population = 10
    generations = 20
    mutation = 0.15
    phases_cnt = 2
    modules_cnt = 6

    logging.basicConfig(level=logging.INFO)

    try:
        opts, args = getopt.getopt(argv, "hp:g:m:", ["pop=", "gen=", "mut=", "phases=", "modules="])
    except getopt.GetoptError:
        logging.error("main.py -p <population (int)> -g <generations (int)> -m <mutation (float)[0:1]> --phases <phases_count (int)[1:10]> --modules <modules_count (int)[1:10]>")
        sys.exit(1)

    for opt, arg in opts:
        if opt == '-h':
            logging.info("main.py -p <population (int)> -g <generations (int)> -m <mutation (float)[0:1]> --phases <phases_count (int)[1:10]> --modules <modules_count (int)[1:10]>")
            sys.exit()
        elif opt in ("-p", "--pop"):
            population = int(arg)
        elif opt in ("-g", "--gen"):
            generations = int(arg)
        elif opt in ("-m", "--mut"):
            mutation = float(arg)
        elif opt == "--phases":
            phases_cnt = int(arg)
        elif opt == "--modules":
            modules_cnt = int(arg)

    pop_size = population
    generations = generations
    tournament_count = 0  # If == 0, Roulette is used
    mutation_probability = mutation
    phases = phases_cnt
    modules = modules_cnt
    genes_cros = True  # Genes crossover
    dataset = 3  # 0 - MNIST, 1 - FASHION MNIST, 2 - SVHN, 3 - CIFAR10, 4 - CIFAR100
    batch_size = 128
    epochs = 10
    val_batch_size = 64
    val_epochs = 50
    val_split = 0.2
    verbose = 1

    result_export = ResultExport(
        pareto_graph=True,
        keras_graph=True,
        keras_model=True,
        genotype_info=True
    )

    ga = NSGAII(
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

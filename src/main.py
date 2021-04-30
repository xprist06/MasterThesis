from evolution_search.nsga_ii import NSGAII
from evolution_search.result_export import ResultExport

import sys
import getopt
import logging


def main(argv):
    pop = None
    gen = None
    mut = None
    phs = None
    mdls = None

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
            pop = int(arg)
        elif opt in ("-g", "--gen"):
            gen = int(arg)
        elif opt in ("-m", "--mut"):
            mut = float(arg)
        elif opt == "--phases":
            phs = int(arg)
        elif opt == "--modules":
            mdls = int(arg)

    pop_size = pop
    generations = gen
    tournament_count = 0  # If == 0, Roulette is used
    mutation_probability = mut
    phases = phs
    modules = mdls
    genes_cros = True  # Genes crossover
    dataset = 3  # 0 - MNIST, 1 - FASHION MNIST, 2 - SVHN, 3 - CIFAR10, 4 - CIFAR100
    batch_size = 128
    epochs = 15
    val_batch_size = 64
    val_epochs = 100
    val_split = 0.2
    verbose = 1

    result_export = ResultExport(
        pareto_graph=True,
        keras_graph=True,
        keras_model=False,
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

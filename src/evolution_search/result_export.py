# -----------------------------------------------------------------------------
# This software was developed as practical part of Master's thesis at FIT BUT
# The program uses multiobjective NSGA-II algorithm for designing accurate
# and compact CNNs.
#
# Author: Jan Pristas, xprist06@stud.fit.vutbr.cz
# Institute: Faculty of Information Technology, Brno University of Technology
#
# File: result_export.py
# Description: Specify which information will be exported
# -----------------------------------------------------------------------------


class ResultExport:
    def __init__(
            self,
            pareto_graph=False,
            keras_graph=False,
            keras_model=False,
            genotype_info=False
    ):
        self.pareto_graph = pareto_graph
        self.keras_graph = keras_graph
        self.keras_model = keras_model
        self.genotype_info = genotype_info

    def export(self):
        """
        Return boolean value if any information should be exported or not
        """
        if self.pareto_graph or \
                self.keras_graph or \
                self.keras_model or \
                self.genotype_info:
            return True
        return False

    def export_individual(self):
        """
        Return boolean value if any information about individual should be exported
        """
        if self.keras_graph or \
                self.keras_model or \
                self.genotype_info:
            return True
        return False

    def export_keras(self):
        """
        Return boolean value if keras graph or model.json should be exported
        """
        if self.keras_graph or \
                self.keras_model:
            return True
        return False

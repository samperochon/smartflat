import os
import sys


from smartflat.constants import available_modality, available_tasks
from smartflat.datasets.loaders import get_dataset
from smartflat.utils.util_coding import *


####################################################
def pff(arg):
    filename = __file__.split("/")[-1]
    print("\033[1m["+filename+"] \033[0m" + arg) 

BOLD = "\033[1m"
CE = "\033[0m"
####################################################

    
class TaskManager():
    """ 
    Manage experiments (processing steps) that should be applied to all the data of a cohort-modality 
    tasks = dict({
        tas_k1 : dict_param_task_1,
        ...
        task_k : dict_param_task_k, 
    })
    """
    def __init__(self, list_tasks_to_do, tasks_parameters, dataset_name = "base", scenario = 'all', task_names='all', modality = "all"):

        self.list_tasks_to_do = list_tasks_to_do
        self.tasks_parameters = tasks_parameters
        self.dataset_name  = dataset_name
        self.scenario = scenario 
        self.task_name = task_names

        if modality == "all" : 
            self.modality = available_modality
        else : 
            self.modality = modality
            
        if task_names == "all" : 
            self.task_names = available_tasks
        else : 
            self.task_names = task_names
            
            
    def do_the_tasks(self):

        dset = get_dataset(
            dataset_name=self.dataset_name,
            scenario=self.scenario,
            modality=self.modality
            )

        # filtered_df = dset.metadata.groupby(['cancer_cohort_from_filename', 'modality_details_from_filename']).first().reset_index()
        # filtered_df = filtered_df[:2]
        # dset.metadata = dset.metadata[:2]
        # blue("{} are going to be done on {} cohort with {} data".format(list(self.tasks.keys()), cohort, modality))
        self.do_tasks(dset=dset)

    def do_tasks(self, dset): 
        for task in self.list_tasks_to_do : 
            tasks_parameters = self.tasks_parameters[task]
            
            if task == "compute_ruptures":
                dset.multiprocess_info_imgs(updates_in_info_getting =tasks_parameters["updates_in_info_getting"] )
                print(task + "done.")
                
            else : 
                raise ValueError

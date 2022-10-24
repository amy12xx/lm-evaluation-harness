"""
LearningQ: A Large-scale Dataset for Educational Question Generation
https://research.monash.edu/en/publications/learningq-a-large-scale-dataset-for-educational-question-generati

LearningQ, a challenging educational question generation dataset 
containing over 230K document-question pairs. It includes 7K 
instructor-designed questions assessing knowledge concepts 
being taught and 223K learner-generated questions seeking 
in-depth understanding of the taught concepts. This is a subset
of the dataset, containing only Khan Academy examples. For the 
full dataset, refer to: https://github.com/AngusGLChen/LearningQ

Homepage: https://github.com/AngusGLChen/LearningQ
"""
from typing import Callable, List, Mapping, Optional, Tuple, Union
import datasets
from lm_eval.api.task import PromptSourceTask


_CITATION = """
@article{learningQ,
    title={LearningQ: A Large-scale Dataset for Educational Question Generation},
    author={Guanliang Chen, Jie Yang, Claudia Hauff, Geert Jan Houben},
    journal={https://aaai.org/ocs/index.php/ICWSM/ICWSM18/paper/view/17857},
    year={2018}
}
"""


class LEARNINGQ(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = None
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def download(self):
        """Downloads and returns the task dataset.

        NOTE: Override this method to download the dataset from a custom API.
        """
        self.dataset = datasets.load_from_disk(
            path=self.DATASET_PATH,
        )
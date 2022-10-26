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
from lm_eval.api.metric import (
    bits_per_byte,
    weighted_perplexity,
)

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
    DATASET_PATH = "learningq_khanacademy"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def download(
        self,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        download_mode: Optional[str] = None,
    ):
        """Downloads and returns the task dataset.

        NOTE: Override this method to download the dataset from a custom API.
        """
        self.dataset = datasets.load_from_disk(
            "/content/" + self.DATASET_PATH,
        )
    
    # def doc_to_text(self, doc):
    #     return "Context: {}\n\nQuestion:".format(
    #         doc["context"]
    #     )
    
    # def doc_to_target(self, doc):
    #     return " {}".format(doc["question"])

    def process_results(
        self, doc: dict, results: list
    ) -> Union[dict, Tuple[dict, dict]]:
        (loglikelihood,) = results
        target = self.doc_to_target(doc)[0]
        words = self.count_words(target)
        bytes_ = self.count_bytes(target)

        out = {
            "word_perplexity": (loglikelihood, words),
            "byte_perplexity": (loglikelihood, bytes_),
            "bits_per_byte": (loglikelihood, bytes_),
        }
        if self.save_examples:
            return out, {
                "word_perplexity_instance": weighted_perplexity(
                    [(loglikelihood, words)]
                ),
                "byte_perplexity_instance": weighted_perplexity(
                    [(loglikelihood, bytes_)]
                ),
                "bits_per_byte_instance": bits_per_byte([(loglikelihood, bytes_)]),
            }
        return out
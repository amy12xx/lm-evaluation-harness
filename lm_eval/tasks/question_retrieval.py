"""
QuestionRetrieval: Question Retrieval dataset based on CS285 video transcripts and questions retrieved based on embeddings and FAISS

"""
import logging
from typing import Callable, List, Mapping, Optional, Tuple, Union
import numpy as np
import datasets
from lm_eval.api.task import PromptSourceTask
from lm_eval.api import utils
from lm_eval.api.metric import (
    bleu,
    mean,
    rouge,
    sari,
    weighted_perplexity,
)

logger = logging.getLogger(__name__)

_CITATION = """
@article{QuestionRetrieval,
    title={QuestionRetrieval: Question Retrieval dataset based on CS285 video transcripts and questions retrieved based on embeddings and FAISS},
    author={Amanda Dsouza},
    journal={TBD},
    year={2022}
}
"""


class QuestionRetrieval(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "question_retrieval"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

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

    def doc_to_target(self, doc):
        return doc["questions"]

    def process_results(
        self, doc: dict, results: list
    ) -> Union[dict, Tuple[dict, dict]]:
        (loglikelihood,) = results
        loglikelihood = loglikelihood.strip().split("\n")[0]

    def process_results(
        self, doc: dict, results: list
    ) -> Union[dict, Tuple[dict, dict]]:
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of sub-metrics and values are the values of
        the metric for that one document.

        NOTE: This function automates processing by using the `promptsource`
        metadata to determine the metric.

        Args:
            doc (dict):
                The document as returned from training_docs, validation_docs, or
                test_docs.
            results (list):
                The results of the requests created in construct_requests.

        Returns:
            A dict of metric results.
        """
        answer_choices_list = self.prompt_template.get_answer_choices_list(doc)
        target = self.doc_to_target(doc)
        if answer_choices_list:
            # If answer_choices_list, then this is a ranked choice prompt.
            # NOTE: In the future, target could be a list of strings.
            assert isinstance(target, list) and len(target) == 1
            target = target[0].strip()
            target_idx = answer_choices_list.index(target)

            pred = answer_choices_list[np.argmax(results)]
            out = {}

            for metric in self.prompt_template.metadata.metrics:
                if metric not in self.CONFIGURED_RANKED_CHOICE_PS_METRICS:
                    logger.warning(
                        f"Unexpected metric: `{metric}`. Add it, or use a task-specific solution."
                    )
                if metric == "Accuracy":
                    out["acc"] = pred == target
                    # Byte-length normalization.
                    completion_len = np.array(
                        [float(len(i)) for i in answer_choices_list]
                    )
                    out["acc_norm"] = (
                        1.0
                        if np.argmax(results / completion_len) == target_idx
                        else 0.0
                    )
            # TODO: Add metrics here.
        else:
            # If not, then this is a generation prompt.
            # NOTE: In the future, target will be a list of strings.
            assert isinstance(target, list)
            pred = results[0].strip().split("\n")[0]
            out = {}
            for metric in self.prompt_template.metadata.metrics:
                if metric not in self.CONFIGURED_GENERATION_PS_METRICS:
                    logger.warning(
                        f"Unexpected metric: `{metric}`. Add it, or use a task-specific solution."
                    )
                if metric == "BLEU":
                    out["bleu"] = (target, pred)
                elif metric == "ROUGE":
                    # TODO: This computes all rouge sub-metrics. Find a generic
                    # way to handle user specified rouge sub-metrics to avoid extra
                    # compute.
                    rouge_scores = rouge(target, pred)
                    # Flatten rouge score dict.
                    rouge_scores = utils.flatten(rouge_scores)
                    # Merge all the rouge-type scores into the `out` dict.
                    out = {**out, **rouge_scores}
                elif metric == "SARI":
                    out["sari"] = sari(self.doc_to_rawtext(doc), pred, target)

        # TODO: Wrap process results s.t. override impl do not
        # override the save examples.
        if self.save_examples:
            example = {
                "pred": pred,
                "target": target,
                "answer_choices_list": answer_choices_list,
            }
            return out, example
        return out
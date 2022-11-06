"""
SQUAD
"""
import datasets
from functools import partial
from math import exp
from packaging import version

from lm_eval.api.request import rf
from lm_eval.api.task import PromptSourceTask


_CITATION = """
@article{Artetxe:etal:2019,
      author    = {Mikel Artetxe and Sebastian Ruder and Dani Yogatama},
      title     = {On the cross-lingual transferability of monolingual representations},
      journal   = {CoRR},
      volume    = {abs/1910.11856},
      year      = {2019},
      archivePrefix = {arXiv},
      eprint    = {1910.11856}
}
"""


def _squad_metric(predictions, references):
    squad_metric = datasets.load_metric("squad_v2")
    return squad_metric.compute(predictions=predictions, references=references)


def _squad_agg(key, items):
    predictions, references = zip(*items)
    return _squad_metric(predictions=predictions, references=references)[key]


class SQUAD(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "squad"
    DATASET_NAME = None

    # HF changed squad on us so we have to make sure we aren't running the old one
    # assert version.parse(datasets.__version__) >= version.parse(
    #     "1.11.0"
    # ), "datasets v1.11.0 or later required for SQuAD"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    # def construct_requests(self, doc: dict, ctx: str, args: dict):
    #     """Uses RequestFactory to construct Requests and returns an iterable of
    #     Requests which will be sent to the LM.

    #     Args:
    #         doc (dict):
    #             The document as returned from training_docs, validation_docs, or
    #             test_docs.
    #         ctx (str):
    #             The context string, generated by fewshot_context. This includes
    #             the natural language description, as well as the few shot examples,
    #             and the question part of the document for `doc`.
    #         args (dict):
    #             The specifics of the context, including number of few shots.

    #     Returns:
    #         An iterable of `Request` objects.
    #     """
    #     request_args = {
    #         "stop_sequences": self.stop_sequences(),
    #         "max_generation_length": self.max_generation_length(),
    #         "num_fewshot": args["num_fewshot"],
    #     }

    #     cont_request = rf.greedy_until(ctx, request_args)
    #     is_unanswerable = rf.loglikelihood(
    #         ctx, self.text_target_separator + "unanswerable"
    #     )

        # return cont_request, is_unanswerable

#     def process_results(self, doc, results):
#         """Take a single document and the LM results and evaluates, returning a
#         dict where keys are the names of sub-metrics and values are the values of
#         the metric for that one document

#         Args:
#             doc (dict):
#                 The document as returned from training_docs, validation_docs, or
#                 test_docs.
#             results (list):
#                 The results of the requests created in construct_requests.

#         Returns:
#             A dict of metric results.
#         """
#         pred, (logprob_unanswerable, _) = results
#         no_answer_probability = exp(logprob_unanswerable)

#         predictions = {
#             "id": doc["id"],
#             "prediction_text": pred,
#             "no_answer_probability": no_answer_probability,
#         }

#         references = {
#             "id": doc["id"],
#             "answers": doc["answers"],
#         }

#         out = {
#             # Exact match (the normalized answer exactly match the gold answer)
#             "exact": (predictions, references),
#             # The F-score of predicted tokens versus the gold answer
#             "f1": (predictions, references),
#             # Exact match (the normalized answer exactly match the gold answer)
#             "HasAns_exact": (predictions, references),
#             # The F-score of predicted tokens versus the gold answer
#             "HasAns_f1": (predictions, references),
#             # No-answer probability threshold associated to the best exact match
#             "best_exact_thresh": (predictions, references),
#             # No-answer probability threshold associated to the best F1
#             "best_f1_thresh": (predictions, references),
#             # Best exact match (with varying threshold)
#             "best_exact": (predictions, references),
#             # Best F1 (with varying threshold)
#             "best_f1": (predictions, references),
#         }
#         if self.save_examples:
#             example = {"pred": pred, "target": doc["answers"]}
#             return out, example
#         return out

#     def aggregation(self):
#         return {
#             # Exact match (the normalized answer exactly match the gold answer)
#             "exact": partial(_squad_agg, "exact"),
#             # The F-score of predicted tokens versus the gold answer
#             "f1": partial(_squad_agg, "f1"),
#             # Exact match (the normalized answer exactly match the gold answer)
#             "HasAns_exact": partial(_squad_agg, "HasAns_exact"),
#             # The F-score of predicted tokens versus the gold answer
#             "HasAns_f1": partial(_squad_agg, "HasAns_f1"),
#             # No-answer probability threshold associated to the best exact match
#             "best_exact_thresh": partial(_squad_agg, "best_exact_thresh"),
#             # No-answer probability threshold associated to the best F1
#             "best_f1_thresh": partial(_squad_agg, "best_f1_thresh"),
#             # Best exact match (with varying threshold)
#             "best_exact": partial(_squad_agg, "best_exact"),
#             # Best F1 (with varying threshold)
#             "best_f1": partial(_squad_agg, "best_f1"),
#         }

#     def higher_is_better(self):
#         return {
#             # Exact match (the normalized answer exactly match the gold answer)
#             "exact": True,
#             "f1": True,  # The F-score of predicted tokens versus the gold answer
#             # Exact match (the normalized answer exactly match the gold answer)
#             "HasAns_exact": True,
#             "HasAns_f1": True,  # The F-score of predicted tokens versus the gold answer
#             # No-answer probability threshold associated to the best exact match
#             "best_exact_thresh": True,
#             "best_f1_thresh": True,  # No-answer probability threshold associated to the best F1
#             "best_exact": True,  # Best exact match (with varying threshold)
#             "best_f1": True,  # Best F1 (with varying threshold)
#         }


import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score


def subject_accuracy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    subject_ids: np.ndarray,
    from_logits: bool = True,
):
    """

    :param y_pred: predicted probabilities of class 1
    :param y_true: ground truth
    :param subject_ids: subject ID
    :param from_logits: if True (default) y_pred is treated as logits and
    therefore softmax is applied in order to transform logits into
    probabilities, otherwise y_pred is treated as probabilities
    :return: subject level accuracy, i.e. a majority vote is taken on the
    predictions for each individual which is then compared against the ground
    truth, in other words a session is considered correctly classified
    if its accuracy is greater than 0.5. subject level accuracy then becomes
    the number of correctly classified  individuals over the total number of
    individuals
    """
    scores = []
    for id in np.unique(subject_ids):
        pred_rec_id = np.where(
            expit(y_pred[subject_ids == id]) > 0.5
            if from_logits
            else y_pred[subject_ids == id] > 0.5,
            1,
            0,
        )
        subject_acc = accuracy_score(
            y_true=y_true[subject_ids == id], y_pred=pred_rec_id
        )
        score = 1 if subject_acc > 0.5 else 0
        scores.append(score)
    return np.mean(scores)


def secondary_metrics_subjects_get_inputs(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    subject_ids: np.ndarray,
):
    subjects_pred, subjects_true, subjects_scores = [], [], []
    for id in np.unique(subject_ids):
        pred_rec_id = np.where(y_pred[subject_ids == id] > 0.5, 1, 0)
        majority_vote = 1 if np.sum(pred_rec_id == 1) > np.sum(pred_rec_id == 0) else 0
        true_rec_id = y_true[subject_ids == id]
        assert len(np.unique(true_rec_id)) == 1
        subjects_pred.append(majority_vote)
        subjects_true.append(true_rec_id.flatten()[0])
        negative_class = 1 - y_pred[subject_ids == id]
        num_segments = len(pred_rec_id)
        subject_probs = np.concatenate(
            (
                negative_class.reshape(num_segments, 1),
                y_pred[subject_ids == id].reshape(num_segments, 1),
            ),
            axis=1,
        )
        # subject_scores is a (1, 2) where [0,0] is the probability of class
        # 0 and [0,1] is the probability of class 1; such probabilities are
        # obtained summing over the class probabilities of each segment in
        # the recording and then normalizing by the number of segments
        subject_scores = np.sum(subject_probs, axis=0) / num_segments
        subjects_scores.append(subject_scores[1])

    return subjects_pred, subjects_true, subjects_scores

import typing as t

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torchmetrics.functional import accuracy
from torchmetrics.functional import mean_squared_error

EPS = torch.finfo(torch.float32).eps


def cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = False,
    reduction: t.Literal["none", "sum", "mean"] = "none",
    eps: t.Union[float, torch.tensor] = EPS,
):
    """cross entropy
    reference: https://github.com/keras-team/keras/blob/e6784e4302c7b8cd116b74a784f4b78d60e83c26/keras/backend.py#L5544
    """
    if from_logits:
        input = F.softmax(input=input, dim=-1)
    p_true_class = torch.clamp(
        input[range(input.shape[0]), target], min=eps, max=1.0 - eps
    )
    loss = -torch.log(p_true_class)
    match reduction:
        case "none":
            pass
        case "mean":
            loss = torch.mean(loss)
        case "sum":
            loss = torch.sum(loss)
        case _:
            raise NotImplementedError(f"loss reduction {reduction} not implemented.")
    return loss


class ReconstructionLoss(_Loss):
    def __init__(
        self,
        args,
    ):
        super(ReconstructionLoss, self).__init__()
        self.channel_names = sorted(args.input_shapes.keys())

    def forward(
        self,
        x_true: t.Dict[str, torch.Tensor],
        x_hat: t.Dict[str, torch.Tensor],
        mask: t.Dict[str, torch.Tensor],
    ):
        channels_reconstruction_loss = 0
        for channel in self.channel_names:
            # Loss is computed on masked (True) entries only
            channel_reconstruction_loss = mean_squared_error(
                target=x_true[channel] * mask[channel],
                preds=x_hat[channel] * mask[channel],
                squared=False,
            )
            channels_reconstruction_loss += channel_reconstruction_loss
        # Divide by num channels to get the channel average rmse
        return channels_reconstruction_loss / len(self.channel_names)


class TFPredictionLoss(_Loss):
    def __init__(
        self,
        args,
    ):
        super(TFPredictionLoss, self).__init__()
        self.register_buffer("eps", torch.tensor(EPS))

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ):
        loss, acc = 0, 0
        for i in range(y_true.shape[-1]):
            loss += cross_entropy(
                input=y_pred[:, i, :],
                target=y_true[:, i],
                from_logits=False,
                reduction="mean",
                eps=self.eps,
            )
            acc += accuracy(
                preds=y_pred[:, i, :],
                target=y_true[:, i],
                task="multiclass",
                num_classes=y_true.shape[-1],
            )
        # Divide by num channels to get the channel average cce and acc
        return loss / y_true.shape[-1], acc / y_true.shape[-1]


class NTXent(_Loss):
    # adapted from: https://theaisummer.com/simclr/
    def __init__(
        self,
        args,
    ):
        super(NTXent, self).__init__()
        self.temperature = args.temperature
        self.device = args.device

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

    def contrastive_accuracy(self, batch_size, similarity_matrix):
        # an entry of the similarity matrix across rows is considered as
        # correctly classified if, given a segment (row), the similarity with
        # the positive sample in that row is higher than that with any negative
        # samples in that row

        # similarity of a segment with itself is not considered, such entries
        # are thus set to -inf
        similarity_matrix[range(batch_size * 2), range(batch_size * 2)] = float("-inf")
        argmax_indices = torch.argmax(similarity_matrix[:batch_size, :], dim=1)
        positive_pair_indices = torch.arange(
            batch_size, batch_size * 2, device=self.device
        )
        return torch.sum(positive_pair_indices == argmax_indices) / batch_size

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        mask = (
            (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool))
            .float()
            .to(self.device)
        )
        z_i = F.normalize(proj_1, p=2, dim=-1)
        z_j = F.normalize(proj_2, p=2, dim=-1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = mask * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)
        acc = self.contrastive_accuracy(
            batch_size=batch_size, similarity_matrix=similarity_matrix
        )
        return loss, acc


class CriticScore(_Loss):
    """
    Critic score
    Further to trying to predict psychometric scale items, the classifier,
    when self.representation_loss_lambda > 0, will try to learn a
    shared-between-tasks representation that will make it harder for the
    critic to tell subjects apart, thus it will try to minimize the probability
    placed on the correct subject for a given segment
    """

    def __init__(self, args):
        super(CriticScore, self).__init__()
        self.register_buffer("coefficient", torch.tensor(args.critic_score_lambda))
        self.register_buffer("one", torch.tensor(1.0))
        self.register_buffer("eps", torch.tensor(EPS))

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        indexes = torch.arange(y_pred.shape[0], device=y_pred.device)
        y_true_prob = y_pred[indexes, y_true]
        loss = torch.mean(-torch.log(self.one - y_true_prob + self.eps))
        return self.coefficient * loss


class CriticLoss(_Loss):
    """
    Cross entropy loss to train critic
    the critic will strive to tell subjects apart from the shared-between-tasks
    representation learned from the classifier's feature_encoder, thus it will
    try to minimize the cross_entropy below
    """

    def __init__(self):
        super(CriticLoss, self).__init__()
        self.register_buffer("eps", torch.tensor(EPS))

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        loss = cross_entropy(
            input=y_pred,
            target=y_true,
            from_logits=False,
            reduction="mean",
            eps=self.eps,
        )
        return loss


class ClassifierLoss(_Loss):
    """
    Cross entropy loss to train critic
    the critic will strive to tell subjects apart from the shared-between-tasks
    representation learned from the classifier's feature_encoder, thus it will
    try to minimize the cross_entropy below
    """

    def __init__(self):
        super(ClassifierLoss, self).__init__()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        loss = F.binary_cross_entropy_with_logits(
            input=y_pred,
            target=y_true,
            reduction="mean",
        )
        return loss


class Criteria:
    """
    Cross entropy loss to train critic
    the critic will strive to tell subjects apart from the shared-between-tasks
    representation learned from the classifier's feature_encoder, thus it will
    try to minimize the cross_entropy below
    """

    def __init__(self, args):
        match args.task_mode:
            case 0:
                match args.pretext_task:
                    case "masked_prediction":
                        self.criterion_sslearner = ReconstructionLoss(args)
                        self.criterion_sslearner.to(args.device)
                    case "transformation_prediction":
                        self.criterion_sslearner = TFPredictionLoss(args)
                        self.criterion_sslearner.to(args.device)
                    case "contrastive":
                        self.criterion_sslearner = NTXent(args)
                        self.criterion_sslearner.to(args.device)
            case 1 | 2 | 3:
                self.critic_score = CriticScore(args)
                self.critic_score.to(args.device)

                self.criterion_critic = CriticLoss()
                self.criterion_critic.to(args.device)

                self.criterion_classifier = ClassifierLoss()
                self.criterion_classifier.to(args.device)


def get_criterion(
    args,
):
    criteria = Criteria(args)

    return criteria

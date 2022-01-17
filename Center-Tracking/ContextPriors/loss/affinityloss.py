import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class AffinityLoss(nn.Layer):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(AffinityLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.num_classes = 150
    
    # ideal_affinity map
    def _construct_ideal_affinity_matrix(self, label, label_size):
        label = paddle.unsqueeze(label, axis=1)
        scaled_labels = F.interpolate(
            label, size=label_size, mode="nearest")

        scaled_labels = scaled_labels.squeeze_().astype('int64').astype("float64")
        scaled_labels[scaled_labels == 255] = self.num_classes
        scaled_labels = scaled_labels.astype('int64')
        one_hot_labels = F.one_hot(scaled_labels, self.num_classes + 1)

        one_hot_labels = one_hot_labels.reshape((one_hot_labels.shape[0],
                                                 -1, self.num_classes + 1))
        
        ideal_affinity_matrix = paddle.bmm(one_hot_labels,
                                           one_hot_labels.transpose((0, 2, 1)))
        return ideal_affinity_matrix

    def forward(self, cls_score, label):
        # print(type(cls_score), cls_score.dtype, type(label), label.dtype)
        ideal_affinity_matrix = self._construct_ideal_affinity_matrix(label, [60, 60]).astype("float64")
        unary_term = F.binary_cross_entropy(cls_score.astype("float64"), ideal_affinity_matrix)

        diagonal_matrix = (1 - paddle.eye(ideal_affinity_matrix.shape[1])).astype("float64")
        vtarget = paddle.multiply(diagonal_matrix, ideal_affinity_matrix)

        recall_part = paddle.sum(cls_score * vtarget.squeeze(), axis=2)
        denominator = paddle.sum(ideal_affinity_matrix, axis=2)
        denominator = paddle.where(denominator <= 0, paddle.ones_like(denominator), denominator)
        recall_part = paddle.divide(recall_part, denominator)
        recall_label = paddle.ones_like(recall_part)
        recall_loss = F.binary_cross_entropy(recall_part, recall_label)

        spec_part = paddle.sum(paddle.multiply(1 - cls_score, 1 - ideal_affinity_matrix), axis=2)
        denominator = paddle.sum(1 - ideal_affinity_matrix, axis=2)
        denominator = paddle.where(denominator <= 0, paddle.ones_like(denominator), denominator)
        spec_part = paddle.divide(spec_part, denominator)
        spec_label = paddle.ones_like(spec_part)
        spec_loss = F.binary_cross_entropy(spec_part, spec_label)

        precision_part = paddle.sum(paddle.multiply(cls_score, ideal_affinity_matrix), axis=2)
        denominator = paddle.sum(cls_score, axis=2)
        denominator = paddle.where(denominator <= 0, paddle.ones_like(denominator), denominator)
        precision_part = paddle.divide(precision_part, denominator)
        precision_label = paddle.ones_like(precision_part)
        precision_loss = F.binary_cross_entropy(precision_part, precision_label)

        global_term = (recall_loss + spec_loss + precision_loss) / 60
        loss_cls = unary_term + global_term
        return loss_cls

# affinityloss = AffinityLoss()
# pred = paddle.rand([2, 3600, 3600])
# la = paddle.randint(0, 150, [2, 560, 560])
# out = affinityloss(pred, la)
# print(out)

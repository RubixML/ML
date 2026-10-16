<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Rubix\ML\Specifications\ProbabilityAndLabelCountsAreEqual;

use function count;
use function array_values;
use function array_keys;
use function asort;

/**
 * ROC AUC
 *
 * The Area Under the Receiver Operating Characteristic (ROC) Curve measures the ability of an
 * estimator to discriminate the positive class from every other class by looking at how well
 * the positive-class probability ranks the samples. A score of 1.0 means that every positive
 * sample has a higher positive-class probability than every negative sample, 0.5 means the
 * estimator is random, and 0.0 means the positive and negative samples are perfectly inverted.
 * If either the positive or negative class is entirely absent from the evaluation set the score
 * is undefined and defaults to 0.5 (chance level).
 *
 * !!! note
 *     Metric assumes probabilities are between 0 and 1 and their joint distribution sums to 1.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ROCAUC implements ProbabilisticMetric
{
    /**
     * The class considered positive when building the ROC curve.
     *
     * @var string|int
     */
    protected string|int $positiveClass;

    /**
     * @param string|int $positiveClass
     */
    public function __construct(string|int $positiveClass)
    {
        $this->positiveClass = $positiveClass;
    }

    /**
     * Return a tuple of the min and max score for this metric.
     *
     * @return Tuple<float,float>
     */
    public function range() : Tuple
    {
        return new Tuple(0.0, 1.0);
    }

    /**
     * Return the validation score of a set of probabilities with their ground-truth labels.
     *
     * @param list<array<string|int,float>> $probabilities
     * @param list<string|int> $labels
     * @return float
     */
    public function score(array $probabilities, array $labels) : float
    {
        ProbabilityAndLabelCountsAreEqual::with($probabilities, $labels)->check();

        $n = count($probabilities);

        if ($n === 0) {
            return 0.0;
        }

        $scores = [];

        foreach ($probabilities as $i => $dist) {
            $scores[$i] = $dist[$this->positiveClass] ?? 0.0;
        }

        return $this->rankSum($scores, $labels);
    }

    /**
     * Compute the AUROC of the samples given their positive-class scores and labels.
     *
     * @param array<int|string,float> $scores
     * @param array<int|string,string|int> $labels
     * @return float
     */
    protected function rankSum(array $scores, array $labels) : float
    {
        $n = count($scores);

        $positives = 0;

        foreach ($labels as $label) {
            if ($label == $this->positiveClass) {
                ++$positives;
            }
        }

        $negatives = $n - $positives;

        if ($positives === 0 or $negatives === 0) {
            return 0.5;
        }

        $sorted = $scores;

        asort($sorted);

        $keys = array_keys($sorted);
        $values = array_values($sorted);

        $ranks = [];

        for ($i = 0, $total = $n; $i < $total; ++$i) {
            $j = $i;
            $value = $values[$i];

            while ($j + 1 < $total and $values[$j + 1] == $value) {
                ++$j;
            }

            $avg = (($i + 1) + ($j + 1)) / 2.0;

            for ($k = $i; $k <= $j; ++$k) {
                $ranks[$keys[$k]] = $avg;
            }

            $i = $j;
        }

        $posSum = 0.0;

        foreach ($labels as $idx => $label) {
            if ($label == $this->positiveClass) {
                $posSum += $ranks[$idx];
            }
        }

        return ($posSum - $positives * ($positives + 1) / 2.0) / ($positives * $negatives);
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "ROCAUC (positive class: {$this->positiveClass})";
    }
}

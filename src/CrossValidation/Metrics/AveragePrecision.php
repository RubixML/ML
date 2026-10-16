<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Rubix\ML\Specifications\ProbabilityAndLabelCountsAreEqual;

use function count;
use function array_values;
use function array_keys;
use function arsort;

/**
 * Average Precision
 *
 * Average Precision measures the ability of an estimator to discriminate the positive class from
 * every other class by looking at how well the positive-class probability ranks the samples. It is
 * equivalent to the Area Under the Precision-Recall Curve. A score of 1.0 means that every positive
 * sample has a higher positive-class probability than every negative sample, and a score of 0.0 can
 * only occur when there are no positive samples. When one or more positive samples are present, the
 * minimum score is determined by their prevalence and is strictly greater than 0, so a fully inverted
 * ranking does not score 0.0.
 *
 * !!! note
 *     Metric assumes probabilities are between 0 and 1 and their joint distribution sums to 1.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class AveragePrecision implements ProbabilisticMetric
{
    /**
     * The class considered positive when computing the average precision.
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

        return $this->areaUnderCurve($scores, $labels);
    }

    /**
     * Compute the area under the precision-recall curve of the samples given their
     * positive-class scores and labels.
     *
     * @param array<int|string,float> $scores
     * @param array<int|string,string|int> $labels
     * @return float
     */
    protected function areaUnderCurve(array $scores, array $labels) : float
    {
        $n = count($scores);

        $positives = 0;

        foreach ($labels as $label) {
            if ($label == $this->positiveClass) {
                ++$positives;
            }
        }

        if ($positives === 0) {
            return 0.0;
        }

        $sorted = $scores;

        arsort($sorted);

        $keys = array_keys($sorted);
        $values = array_values($sorted);

        $total = $n;

        $cumPositives = 0.0;
        $cumPredicted = 0.0;
        $prevRecall = 0.0;

        $area = 0.0;

        for ($i = 0; $i < $total;) {
            $value = $values[$i];
            $end = $i;

            while ($end + 1 < $total and $values[$end + 1] == $value) {
                ++$end;
            }

            $groupPositives = 0;

            for ($k = $i; $k <= $end; ++$k) {
                if ($labels[$keys[$k]] == $this->positiveClass) {
                    ++$groupPositives;
                }
            }

            $cumPositives += $groupPositives;
            $cumPredicted += $end - $i + 1;

            $precision = $cumPositives / $cumPredicted;
            $recall = $cumPositives / $positives;

            $area += ($recall - $prevRecall) * $precision;

            $prevRecall = $recall;

            $i = $end + 1;
        }

        return $area;
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return "AveragePrecision (positive class: {$this->positiveClass})";
    }
}

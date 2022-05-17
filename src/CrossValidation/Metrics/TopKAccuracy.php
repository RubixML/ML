<?php

namespace Rubix\ML\CrossValidation\Metrics;

use Rubix\ML\Tuple;
use Rubix\ML\Specifications\ProbabilityAndLabelCountsAreEqual;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function array_keys;
use function array_slice;
use function arsort;
use function count;

/**
 * Top K Accuracy
 *
 * Top K Accuracy looks at the k classes with the highest predicted probabilities when
 * calculating the accuracy score. If one of the top k classes matches the ground-truth,
 * then the prediction is considered accurate.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class TopKAccuracy implements ProbabilisticMetric
{
    /**
     * The number of classes with the highest predicted probability to consider.
     *
     * @var int
     */
    protected int $k;

    /**
     * @param int $k
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $k = 3)
    {
        if ($k < 1) {
            throw new InvalidArgumentException('K must be'
                . " greater than 0, $k given.");
        }

        $this->k = $k;
    }

    /**
     * {@inheritDoc}
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

        $score = 0.0;

        foreach ($probabilities as $i => $dist) {
            $label = $labels[$i];

            arsort($dist);

            $topClasses = array_keys(array_slice($dist, 0, $this->k, true));

            foreach ($topClasses as $class) {
                if ($class == $label) {
                    ++$score;

                    break;
                }
            }
        }

        return $score / $n;
    }

    /**
     * {@inheritDoc}
     */
    public function __toString() : string
    {
        return "Top K Accuracy (k: {$this->k})";
    }
}

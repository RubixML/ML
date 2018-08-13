<?php

namespace Rubix\ML\CrossValidation;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Labeled;
use MathPHP\Statistics\Average;
use Rubix\ML\CrossValidation\Metrics\Metric;
use InvalidArgumentException;

class LeavePOut implements Validator
{
    /**
     * The number of samples to leave out each round for testing.
     *
     * @var int
     */
    protected $p;

    /**
     * @param  int  $p
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $p = 10)
    {
        if ($p < 1) {
            throw new InvalidArgumentException('The number of held out samples'
                . ' must be 1 or more.');
        }

        $this->p = $p;
    }

    /**
     * @param  \Rubix\ML\Estimator  $estimator
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @param  \Rubix\ML\CrossValidation\Metrics\Metric  $metric
     * @return float
     */
    public function test(Estimator $estimator, Labeled $dataset, Metric $metric) : float
    {
        $n = (int) round($dataset->numRows() / $this->p);

        $dataset->randomize();

        $scores = [];

        for ($i = 0; $i < $n; $i++) {
            $training = clone $dataset;

            $testing = $training->splice($i * $this->p, $this->p);

            $estimator->train($training);

            $scores[] = $metric->score($estimator, $testing);
        }

        return Average::mean($scores);
    }
}

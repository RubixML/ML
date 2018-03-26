<?php

namespace Rubix\Engine;

use Rubix\Engine\Math\Stats;

class Test
{
    /**
     * The estimator being tested.
     *
     * @var \Rubix\Engine\Estimator
     */
    protected $estimator;

    /**
     * @param  \Rubix\Engine\Estimator  $estimator
     * @return void
     */
    public function __construct(Estimator $estimator)
    {
        $this->estimator = $estimator;
    }

    /**
     * Calculate the accuracy of the estimator.
     *
     * @return float
     */
    public function accuracy(array $samples, array $outcomes, float $precision = 5) : float
    {
        $score = 0;

        foreach ($samples as $i => $sample) {
            $prediction = $this->estimator->predict($sample);

            if ($prediction['outcome'] === $outcomes[$i]) {
                $score++;
            }
        }

        return Stats::round(($score / count($samples)) * 100, $precision);
    }
}

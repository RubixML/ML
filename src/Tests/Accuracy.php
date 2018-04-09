<?php

namespace Rubix\Engine\Tests;

use MathPHP\Statistics\Average;
use MathPHP\Statistics\Descriptive;
use InvalidArgumentException;
use RuntimeException;

class Accuracy implements Test
{
    /**
     * The minimum accuracy score to pass the test.
     *
     * @var int
     */
    protected $threshold;

    /**
     * @param  float  $threshold
     * @return void
     */
    public function __construct(float $threshold = 0.9)
    {
        if ($threshold < 0.0 || $threshold > 1.0) {
            throw new InvalidArgumentException('Minimum accuracy must be a float value between 0 and 1.');
        }

        $this->threshold = $threshold;
    }

    /**
     * Test the accuracy of the estimator.
     *
     * @param  array  $predictions
     * @param  array|null  $outcomes
     * @throws \InvalidArgumentException
     * @return bool
     */
    public function test(array $predictions, ?array $outcomes = null) : bool
    {
        if (!isset($outcomes)) {
            throw new RuntimeException('This test requires the labeled outcomes of a supervised dataset.');
        }

        if (count($predictions) < 1) {
            throw new RuntimeException('This test requires at least 1 prediction.');
        }

        $accuracy = 0.0;

        if (reset($predictions)->categorical()) {
            $accuracy = $this->calculateCategoricalAccuracy($predictions, $outcomes);
        } else if (reset($predictions)->continuous()) {
            $accuracy = $this->calculateContinuousAccuracy($predictions, $outcomes);
        }

        $pass = $accuracy >= $this->threshold;

        echo 'Model is ' . (string) round($accuracy * 100, 3) . '% accurate - ' . ($pass ? 'PASS' : 'FAIL') . "\n";

        return $pass;
    }

    /**
     * Calculate the accuracy of a classifier.
     *
     * @param  array  $predictions
     * @param  array  $outcomes
     * @return float
     */
    protected function calculateCategoricalAccuracy(array $predictions, array $outcomes) : float
    {
        $score = 0;

        foreach ($predictions as $i => $prediction) {
            if ($prediction->outcome() === $outcomes[$i]) {
                $score++;
            }
        }

        return $score / count($outcomes);
    }

    /**
     * Calculate the accuracy of a regression.
     *
     * @param  array  $predictions
     * @param  array  $outcomes
     * @return float
     */
    protected function calculateContinuousAccuracy(array $predictions, array $outcomes) : float
    {
        $errors = [];

        foreach ($predictions as $i => $prediction) {
            $errors[] = ($outcomes[$i] - $prediction->outcome()) ** 2;
        }

        return (1 / count($errors)) * array_sum($errors);
    }
}

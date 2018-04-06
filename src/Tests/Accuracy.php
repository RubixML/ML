<?php

namespace Rubix\Engine\Tests;

use Rubix\Engine\Classifier;
use Rubix\Engine\Regression;
use MathPHP\Statistics\Average;
use MathPHP\Statistics\Descriptive;
use Rubix\Engine\SupervisedDataset;
use InvalidArgumentException;

class Accuracy extends Test
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
     * @param \Rubix\Engine\SupervisedDataset  $data
     * @return bool
     */
    public function test(SupervisedDataset $data) : bool
    {
        $outcomes = $data->outcomes();
        $predictions = [];
        $accuracy = 0;

        foreach ($data->samples() as $sample) {
            $predictions[] = $this->estimator->predict($sample)['outcome'];
        }

        if ($this->estimator instanceof Classifier) {
            $accuracy = $this->calculateCategoricalAccuracy($predictions, $outcomes);
        } else if ($this->estimator instanceof Regression) {
            $accuracy = $this->calculateContinuousAccuracy($predictions, $outcomes);
        }

        $pass = $accuracy >= $this->threshold;

        echo 'Model is ' . (string) round($accuracy * 100, 5) . '% accurate - ' . ($pass ? 'PASS' : 'FAIL') . "\n";

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
            if ($prediction === $outcomes[$i]) {
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
            $errors[] = ($outcomes[$i] - $prediction) ** 2;
        }

        return 1 - sqrt(Average::mean($errors)) / Descriptive::standardDeviation($errors);
    }
}

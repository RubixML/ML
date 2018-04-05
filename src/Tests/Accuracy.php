<?php

namespace Rubix\Engine\Tests;

use Rubix\Engine\Classifier;
use Rubix\Engine\Regression;
use MathPHP\Statistics\Average;
use Rubix\Engine\SupervisedDataset;
use InvalidArgumentException;

class Accuracy extends Test
{
    /**
     * The minimum accuracy score to pass the test.
     *
     * @var int
     */
    protected $accuracy;

    /**
     * The decimal precision of the accuracy measurement.
     *
     * @var int
     */
    protected $precision;

    /**
     * @param  float  $accuracy
     * @param  int  $precision
     * @return void
     */
    public function __construct(float $accuracy = 90.0, int $precision = 4)
    {
        if ($accuracy < 0 || $accuracy > 100) {
            throw new InvalidArgumentException('Minimum accuracy must be a float value between 0 and 100.');
        }

        $this->accuracy = $accuracy;
        $this->precision = $precision;
    }

    /**
     * Test the accuracy of the estimator.
     *
     * @param \Rubix\Engine\SupervisedDataset  $data
     * @return bool
     */
    public function test(SupervisedDataset $data) : bool
    {
        $accuracy = 0;

        if ($this->estimator instanceof Classifier) {
            $outcomes = $data->outcomes();
            $score = 0;

            foreach ($data->samples() as $i => $sample) {
                $prediction = $this->estimator->predict($sample);

                if ($prediction['outcome'] === $outcomes[$i]) {
                    $score++;
                }
            }

            $accuracy = round(($score / $data->count()) * 100, $this->precision);
        }

        $pass = $accuracy >= $this->accuracy;

        echo 'Model is ' . (string) $accuracy . '% accurate - ' . ($pass ? 'PASS' : 'FAIL') . "\n";

        return $pass;
    }
}

<?php

namespace Rubix\Engine;

use Rubix\Engine\Tests\Test;
use MathPHP\Statistics\Average;

class Prototype
{
    /**
     * The estimator.
     *
     * @var \Rubix\Engine\Estimator
     */
    protected $estimator;

    /**
     * The testing middleware stack.
     *
     * @var array
     */
    protected $tests = [
        //
    ];

    /**
     * @param  \Rubix\Engine\Estimator  $estimator
     * @param  array  $tests
     * @return void
     */
    public function __construct(Estimator $estimator, array $tests = [])
    {
        foreach ($tests as $test) {
            $this->addTest($test);
        }

        $this->estimator = $estimator;
    }

    /**
     * @param  \Rubix\Engine\Dataset  $data
     * @return void
     */
    public function train(Dataset $data) : void
    {
        $start = microtime(true);

        $this->estimator->train($data);

        $timestamp = microtime(true) - $start;

        echo 'Training completed in ' . (string) round($timestamp, 5) . 's' . "\n";
    }

    /**
     * @param  array  $sample
     * @return \Rubix\Engine\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        $start = microtime(true);

        $prediction = $this->estimator->predict($sample);

        $timestamp = microtime(true) - $start;

        return $prediction->addMeta([
            'prediction_time' => $timestamp,
        ]);
    }

    /**
     * Run the tests on the estimator.
     *
     * @param  \Rubix\Engine\Dataset  $data
     * @return bool
     */
    public function test(Dataset $data) : bool
    {
        $predictions = $timestamps = [];

        foreach ($data->samples() as $sample) {
            $prediction = $this->predict($sample);

            $predictions[] = $prediction;
            $timestamps[] = $prediction->meta('prediction_time');
        }

        $results = array_map(function ($test) use ($predictions, $data) {
            return $test->test($predictions, $data->outcomes());
        }, $this->tests);

        $pass = !in_array(false, $results);

        echo 'Average prediction took ' . (string) round(Average::mean($timestamps), 5) . 's' . "\n";
        echo 'Testing finished - ' . ($pass ? 'PASS' : 'FAIL') . "\n";

        return $pass;
    }

    /**
     * Add a test to the testing stack.
     *
     * @param  \Rubix\Engine\Tests\Test  $test
     * @return self
     */
    public function addTest(Test $test) : self
    {
        $this->tests[] = $test;

        return $this;
    }
}

<?php

namespace Rubix\Engine;

use Rubix\Engine\Tests\Test;
use Rubix\Engine\Tests\Loggers\Screen;
use Rubix\Engine\Tests\Loggers\Logger;

class Prototype implements Estimator
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
     * The logger used to log messages emitted by the model.
     *
     * @var \Rubix\Engine\Loggers\Logger
     */
    protected $logger;

    /**
     * @param  \Rubix\Engine\Estimator  $estimator
     * @param  array  $tests
     * @param  \Runix\Engine\Loggers\Logger  $logger
     * @return void
     */
    public function __construct(Estimator $estimator, array $tests = [], Logger $logger = null)
    {
        if (!isset($logger)) {
            $logger = new Screen();
        }

        $this->estimator = $estimator;
        $this->logger = $logger;

        foreach ($tests as $test) {
            $this->addTest($test);
        }
    }

    /**
     * Return the underlying estimator instance.
     *
     * @return \Rubix\Engine\Estimator
     */
    public function estimator() : Estimator
    {
        return $this->estimator;
    }

    /**
     * @param  \Rubix\Engine\Dataset  $data
     * @return void
     */
    public function train(Dataset $data) : void
    {
        $this->logger->log('Training started');

        $start = microtime(true);

        $this->estimator->train($data);

        $timestamp = microtime(true) - $start;

        $this->logger->log('Training completed in ' . (string) round($timestamp, 5) . 's');
    }

    /**
     * @param  array  $sample
     * @return \Rubix\Engine\Prediction
     */
    public function predict(array $sample) : Prediction
    {
        return $this->estimator->predict($sample);
    }

    /**
     * Run the tests on the estimator.
     *
     * @param  \Rubix\Engine\SupervisedDataset  $data
     * @return bool
     */
    public function test(SupervisedDataset $data) : bool
    {
        $this->logger->log('Testing started');

        $start = microtime(true);

        $predictions = array_map(function ($sample) {
            return $this->predict($sample)->outcome();
        }, $data->samples());

        $timestamp = (microtime(true) - $start) / count($data);

        $results = array_map(function ($test) use ($predictions, $data) {
            return $test->score($predictions, $data->outcomes());
        }, $this->tests);

        $this->logger->log('Model took ' . (string) round($timestamp, 5) . 's on average to make a prediction.');
        $this->logger->log('Testing completed');

        return true;
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

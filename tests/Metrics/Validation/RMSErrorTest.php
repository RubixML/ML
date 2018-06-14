<?php

use Rubix\ML\Datasets\Labeled;
use Rubix\Tests\Helpers\MockRegressor;
use Rubix\ML\Metrics\Validation\RMSError;
use Rubix\ML\Metrics\Validation\Validation;
use Rubix\ML\Metrics\Validation\Regression;
use PHPUnit\Framework\TestCase;

class RMSErrorTest extends TestCase
{
    protected $metric;

    public function setUp()
    {
        $this->testing = new Labeled([[], [], [], [], []], [10, 10, 6, 14, 8]);

        $this->estimator = new MockRegressor([9, 15, 9, 12, 8]);

        $this->metric = new RMSError();
    }

    public function test_build_rms_error_metric()
    {
        $this->assertInstanceOf(RMSError::class, $this->metric);
        $this->assertInstanceOf(Regression::class, $this->metric);
        $this->assertInstanceOf(Validation::class, $this->metric);
    }

    public function test_score_predictions()
    {
        $score = $this->metric->score($this->estimator, $this->testing);

        $this->assertEquals(2.2, $score, '', 5);
    }
}

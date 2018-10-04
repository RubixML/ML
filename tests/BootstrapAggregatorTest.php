<?php

namespace Rubix\ML\Tests;

use Rubix\ML\Ensemble;
use Rubix\ML\Persistable;
use Rubix\ML\MetaEstimator;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\BootstrapAggregator;
use Rubix\ML\Regressors\RegressionTree;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class BootstrapAggregatorTest extends TestCase
{
    const TOLERANCE = 3.5;

    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $this->training = Labeled::load(__DIR__ . '/mpg.dataset');

        $this->testing = $this->training->randomize()->head(3);

        $this->estimator = new BootstrapAggregator(RegressionTree::class, [10, 3, 2], 10, 0.8);
    }

    public function test_build_meta_estimator()
    {
        $this->assertInstanceOf(BootstrapAggregator::class, $this->estimator);
        $this->assertInstanceOf(MetaEstimator::class, $this->estimator);
        $this->assertInstanceOf(Ensemble::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    public function test_make_prediction()
    {
        $this->estimator->train($this->training);

        $predictions = $this->estimator->predict($this->testing);

        $this->assertEquals($this->testing->label(0), $predictions[0], '', self::TOLERANCE);
        $this->assertEquals($this->testing->label(1), $predictions[1], '', self::TOLERANCE);
        $this->assertEquals($this->testing->label(2), $predictions[2], '', self::TOLERANCE);
    }

    public function test_predict_untrained()
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict($this->testing);
    }
}

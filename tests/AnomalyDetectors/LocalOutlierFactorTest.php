<?php

namespace Rubix\ML\Tests\AnomalyDetectors;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\AnomalyDetectors\LocalOutlierFactor;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class LocalOutlierFactorTest extends TestCase
{
    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $this->training = Labeled::load(dirname(__DIR__) . '/iris.dataset');

        $this->testing = new Labeled([
            [6.9, 3.2, 5.7, 2.3], [6.4, 3.1, 5.5, 1.8], [5.5, 2.4, 3.8, 1.1],
            [6.8, 3.2, 5.9, 2.3], [5.7, 3.8, 1.7, 0.3], [5.4, 3.9, 1.7, 0.4],
        ], [1, 1, 0, 1, 0, 0]);

        $this->estimator = new LocalOutlierFactor(10, 10, 0.6, new Euclidean());
    }

    public function test_build_detector()
    {
        $this->assertInstanceOf(LocalOutlierFactor::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Online::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::DETECTOR, $this->estimator->type());
    }

    public function test_make_prediction()
    {
        $this->estimator->train($this->training);

        $predictions = $this->estimator->predict($this->testing);

        $this->assertEquals($this->testing->label(0), $predictions[0]);
        $this->assertEquals($this->testing->label(1), $predictions[1]);
        $this->assertEquals($this->testing->label(2), $predictions[2]);
        $this->assertEquals($this->testing->label(3), $predictions[3]);
        $this->assertEquals($this->testing->label(4), $predictions[4]);
        $this->assertEquals($this->testing->label(5), $predictions[5]);
    }

    public function test_partial_train()
    {
        $folds = $this->training->randomize()->fold(2);

        $this->estimator->train($folds[0]);

        $this->estimator->partial($folds[1]);

        $predictions = $this->estimator->predict($this->testing);

        $this->assertEquals($this->testing->label(0), $predictions[0]);
        $this->assertEquals($this->testing->label(1), $predictions[1]);
        $this->assertEquals($this->testing->label(2), $predictions[2]);
        $this->assertEquals($this->testing->label(3), $predictions[3]);
        $this->assertEquals($this->testing->label(4), $predictions[4]);
        $this->assertEquals($this->testing->label(5), $predictions[5]);
    }

    public function test_predict_untrained()
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict($this->testing);
    }
}

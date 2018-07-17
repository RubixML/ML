<?php

namespace Rubix\Tests\Classifiers;

use Rubix\ML\Ensemble;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\Binary;
use Rubix\ML\Classifiers\AdaBoost;
use Rubix\ML\Classifiers\ExtraTree;
use Rubix\ML\Classifiers\Classifier;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class AdaBoostTest extends TestCase
{
    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $this->training = Labeled::restore(dirname(__DIR__) . '/iris.dataset');

        $this->testing = $this->training->randomize()->head(3);

        $this->estimator = new AdaBoost(ExtraTree::class, [10, 3, 4], 100, 0.2, 1e-3);
    }

    public function test_build_classifier()
    {
        $this->assertInstanceOf(AdaBoost::class, $this->estimator);
        $this->assertInstanceOf(Classifier::class, $this->estimator);
        $this->assertInstanceOf(Binary::class, $this->estimator);
        $this->assertInstanceOf(Ensemble::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    public function test_make_prediction()
    {
        $this->training->randomize();

        $this->estimator->train($this->training);

        $predictions = $this->estimator->predict($this->testing);

        $this->assertEquals($this->testing->label(0), $predictions[0]);
        $this->assertEquals($this->testing->label(1), $predictions[1]);
        $this->assertEquals($this->testing->label(2), $predictions[2]);
    }

    public function test_train_with_unlabeled()
    {
        $dataset = new Unlabeled([['bad']]);

        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train($dataset);
    }
}

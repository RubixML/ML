<?php

namespace Rubix\ML\Tests\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Ensemble;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\AdaBoost;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class AdaBoostTest extends TestCase
{
    const TRAIN_SIZE = 300;
    const TEST_SIZE = 5;

    protected $generator;

    protected $estimator;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            'red' => new Blob([255, 0, 0], 3.),
            'green' => new Blob([0, 128, 0], 1.),
            'blue' => new Blob([0, 0, 255], 2.),
        ]);

        $this->estimator = new AdaBoost(new ClassificationTree(3), 150, 1., 0.8, 1e-4);
    }

    public function test_build_classifier()
    {
        $this->assertInstanceOf(AdaBoost::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Ensemble::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::CLASSIFIER, $this->estimator->type());
    }

    public function test_train_predict()
    {
        $this->estimator->train($this->generator->generate(self::TRAIN_SIZE));

        $testing = $this->generator->generate(self::TEST_SIZE);

        foreach ($this->estimator->predict($testing) as $i => $prediction) {
            $this->assertEquals($testing->label($i), $prediction);
        }
    }

    public function test_train_with_unlabeled()
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Unlabeled::quick());
    }

    public function test_predict_untrained()
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

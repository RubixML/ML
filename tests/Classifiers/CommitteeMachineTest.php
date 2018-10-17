<?php

namespace Rubix\ML\Tests\Classifiers;

use Rubix\ML\Ensemble;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\Classifiers\CommitteeMachine;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class CommitteeMachineTest extends TestCase
{
    const TRAIN_SIZE = 150;
    const TEST_SIZE = 5;
    const MIN_PROB = 0.33;

    protected $generator;

    protected $estimator;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            'inner' => new Circle(0., 0., 1., 0.01),
            'middle' => new Circle(0., 0., 5., 0.05),
            'outer' => new Circle(0., 0., 10., 0.1),
        ]);

        $this->estimator = new CommitteeMachine([
            new ClassificationTree(10, 3, 2),
            new KNearestNeighbors(3, new Euclidean()),
            new GaussianNB(),
        ]);
    }

    public function test_build_classifier()
    {
        $this->assertInstanceOf(CommitteeMachine::class, $this->estimator);
        $this->assertInstanceOf(Ensemble::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::CLASSIFIER, $this->estimator->type());
    }

    public function test_train_predict_proba()
    {
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($this->generator->generate(self::TRAIN_SIZE));

        foreach ($this->estimator->predict($testing) as $i => $prediction) {
            $this->assertEquals($testing->label($i), $prediction);
        }

        foreach ($this->estimator->proba($testing) as $i => $prob) {
            $this->assertGreaterThan(self::MIN_PROB, $prob[$testing->label($i)]);
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

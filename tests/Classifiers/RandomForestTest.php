<?php

namespace Rubix\ML\Tests\Classifiers;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class RandomForestTest extends TestCase
{
    const TRAIN_SIZE = 350;
    const TEST_SIZE = 10;
    const MIN_SCORE = 0.9;

    protected $generator;

    protected $estimator;

    protected $metric;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            'red' => new Blob([255, 0, 0], 3.),
            'green' => new Blob([0, 128, 0], 1.),
            'blue' => new Blob([0, 0, 255], 2.),
        ], [3, 4, 3]);

        $this->estimator = new RandomForest(new ClassificationTree(10), 100, 0.2);

        $this->metric = new Accuracy();
    }

    public function test_build_classifier()
    {
        $this->assertInstanceOf(RandomForest::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::CLASSIFIER, $this->estimator->type());
    }

    public function test_train_predict()
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
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

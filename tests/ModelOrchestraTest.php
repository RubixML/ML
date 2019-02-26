<?php

namespace Rubix\ML\Tests;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\ModelOrchestra;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Other\Loggers\BlackHole;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\F1Score;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class ModelOrchestraTest extends TestCase
{
    protected const TRAIN_SIZE = 300;
    protected const TEST_SIZE = 10;
    protected const MIN_SCORE = 0.9;

    protected $generator;

    protected $estimator;

    protected $metric;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            'inner' => new Circle(0., 0., 1., 0.01),
            'middle' => new Circle(0., 0., 5., 0.05),
            'outer' => new Circle(0., 0., 10., 0.1),
        ], [3, 3, 4]);

        $this->estimator = new ModelOrchestra([
            new ClassificationTree(10, 3, 2),
            new KNearestNeighbors(3, new Euclidean()),
            new GaussianNB(),
        ], new SoftmaxClassifier(10), 0.8);

        $this->metric = new F1Score();

        $this->estimator->setLogger(new BlackHole());
    }

    public function test_build_classifier()
    {
        $this->assertInstanceOf(ModelOrchestra::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);

        $this->assertEquals(Estimator::CLASSIFIER, $this->estimator->type());

        $this->assertNotContains(DataType::CATEGORICAL, $this->estimator->compatibility());
        $this->assertContains(DataType::CONTINUOUS, $this->estimator->compatibility());
    }

    public function test_train_predict()
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    public function test_train_with_unlabeled()
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Unlabeled::quick());
    }
}

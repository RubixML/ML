<?php

declare(strict_types=1);

namespace Rubix\ML\Tests;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\CommitteeMachine;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Tests\DataProvider\BackendProviderTrait;

#[Group('MetaEstimators')]
#[CoversClass(CommitteeMachine::class)]
class CommitteeMachineTest extends TestCase
{
    use BackendProviderTrait;

    protected const int TRAIN_SIZE = 512;

    protected const int TEST_SIZE = 256;

    protected const float MIN_SCORE = 0.9;

    protected const int RANDOM_SEED = 0;

    protected Agglomerate $generator;

    protected CommitteeMachine $estimator;

    protected Accuracy $metric;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'inner' => new Circle(x: 0.0, y: 0.0, scale: 1.0, noise: 0.01),
                'middle' => new Circle(x: 0.0, y: 0.0, scale: 5.0, noise: 0.05),
                'outer' => new Circle(x: 0.0, y: 0.0, scale: 10.0, noise: 0.15),
            ],
            weights: [3, 3, 4]
        );

        $this->estimator = new CommitteeMachine(
            experts: [
                new ClassificationTree(maxHeight: 10, maxLeafSize: 3, minPurityIncrease: 2),
                new KNearestNeighbors(k: 3),
                new GaussianNB(),
            ],
            influences: [3, 4, 5]
        );

        $this->metric = new Accuracy();

        srand(self::RANDOM_SEED);
    }

    public function testAssertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    public function testType() : void
    {
        $this->assertEquals(EstimatorType::classifier(), $this->estimator->type());
    }

    public function testCompatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    public function testParams() : void
    {
        $expected = [
            'experts' => [
                new ClassificationTree(maxHeight: 10, maxLeafSize: 3, minPurityIncrease: 2),
                new KNearestNeighbors(k: 3),
                new GaussianNB(),
            ],
            'influences' => [
                0.25,
                0.3333333333333333,
                0.4166666666666667,
            ],
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    /**
     * @param Backend $backend
     */
    #[DataProvider('provideBackends')]
    public function testTrainPredict(Backend $backend) : void
    {
        $this->estimator->setBackend($backend);

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        /** @var list<int> $predictions */
        $predictions = $this->estimator->predict($testing);

        /** @var list<int|string> $labels */
        $labels = $testing->labels();
        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    public function testTrainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Unlabeled::quick(samples: [['bad']]));
    }

    public function testPredictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Regressors;

use Generator;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Graph\Trees\Spatial;
use Rubix\ML\Graph\Trees\VantageTree;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\HalfMoon;
use Rubix\ML\Regressors\KDNeighborsRegressor;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Regressors')]
#[CoversClass(KDNeighborsRegressor::class)]
class KDNeighborsRegressorTest extends TestCase
{
    /**
     * The number of samples in the training set.
     */
    protected const int TRAIN_SIZE = 512;

    /**
     * The number of samples in the validation set.
     */
    protected const int TEST_SIZE = 256;

    /**
     * The minimum validation score required to pass the test.
     */
    protected const MIN_SCORE = 0.89;

    /**
     * Constant used to see the random number generator.
     */
    protected const int RANDOM_SEED = 0;

    protected HalfMoon $generator;

    protected KDNeighborsRegressor $estimator;

    protected RSquared $metric;

    public static function trainPredictProvider() : Generator
    {
        yield 'kd tree' => [new KDTree()];
        yield 'ball tree' => [new BallTree()];
        yield 'vantage tree' => [new VantageTree()];
    }

    protected function setUp() : void
    {
        $this->generator = new HalfMoon(x: 4.0, y: -7.0, scale: 1.0, rotation: 90, noise: 0.25);

        $this->estimator = new KDNeighborsRegressor(k: 5, weighted: true, tree: new KDTree());

        $this->metric = new RSquared();

        srand(self::RANDOM_SEED);
    }

    #[Test]
    public function preConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    #[Test]
    public function badK() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new KDNeighborsRegressor(k: 0);
    }

    #[Test]
    public function type() : void
    {
        $this->assertEquals(EstimatorType::regressor(), $this->estimator->type());
    }

    #[Test]
    public function compatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    #[Test]
    public function params() : void
    {
        $expected = [
            'k' => 5,
            'weighted' => true,
            'tree' => new KDTree(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    #[DataProvider('trainPredictProvider')]
    #[Test]
    public function trainPredict(Spatial $tree) : void
    {
        $estimator = new KDNeighborsRegressor(k: 5, weighted: true, tree: $tree);

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $estimator->train($training);

        $this->assertTrue($estimator->trained());

        $predictions = $estimator->predict($testing);

        /** @var list<int|float> $labels */
        $labels = $testing->labels();

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick(samples: [['bad']], labels: [2]));
    }

    #[Test]
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

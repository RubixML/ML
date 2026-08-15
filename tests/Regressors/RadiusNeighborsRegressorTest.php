<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Regressors;

use Generator;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\Datasets\Generators\HalfMoon;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Regressors\RadiusNeighborsRegressor;

#[Group('Regressors')]
#[CoversClass(RadiusNeighborsRegressor::class)]
class RadiusNeighborsRegressorTest extends TestCase
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
    protected const float MIN_SCORE = 0.9;

    /**
     * Constant used to see the random number generator.
     */
    protected const int RANDOM_SEED = 0;

    protected HalfMoon $generator;

    protected RadiusNeighborsRegressor $estimator;

    protected RSquared $metric;

    public static function predictionChecks() : Generator
    {
        yield 'default dataset sizes' => [self::TRAIN_SIZE, self::TEST_SIZE];
    }

    protected function setUp() : void
    {
        $this->generator = new HalfMoon(x: 4.0, y: -7.0, scale: 1.0, rotation: 90, noise: 0.25);

        $this->estimator = new RadiusNeighborsRegressor(radius: 0.8, weighted: true, tree: new BallTree());

        $this->metric = new RSquared();

        srand(self::RANDOM_SEED);
    }

    #[Test]
    #[TestDox('Estimator is untrained before fitting')]
    public function testAssertPreConditions() : void
    {
        self::assertFalse($this->estimator->trained());
    }

    #[Test]
    #[TestDox('Radius must be greater than zero')]
    public function badRadius() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RadiusNeighborsRegressor(radius: 0.0);
    }

    #[Test]
    #[TestDox('Estimator type is regressor')]
    public function type() : void
    {
        self::assertEquals(EstimatorType::regressor(), $this->estimator->type());
    }

    #[Test]
    #[TestDox('Compatibility only includes continuous data')]
    public function compatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        self::assertEquals($expected, $this->estimator->compatibility());
    }

    #[Test]
    #[TestDox('It trains and predicts with the expected score')]
    public function trainPredict() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        self::assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict($testing);

        /** @var list<int|float> $labels */
        $labels = $testing->labels();
        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        self::assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    #[TestDox('Predictions match the test set and remain finite')]
    #[DataProvider('predictionChecks')]
    public function trainPredictChecks(int $trainSize, int $testSize) : void
    {
        $training = $this->generator->generate($trainSize);
        $testing = $this->generator->generate($testSize);

        $this->estimator->train($training);

        $predictions = $this->estimator->predict($testing);

        self::assertCount($testSize, $predictions);

        foreach ($predictions as $prediction) {
            self::assertIsFloat($prediction);
            self::assertFalse(is_nan($prediction));
        }

        /** @var list<int|float> $labels */
        $labels = $testing->labels();
        $score = $this->metric->score(predictions: $predictions, labels: $labels);

        self::assertIsFloat($score);
        self::assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    #[TestDox('Training rejects incompatible labels')]
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick(samples: [['bad']], labels: [2]));
    }

    #[Test]
    #[TestDox('Predicting before training throws an exception')]
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

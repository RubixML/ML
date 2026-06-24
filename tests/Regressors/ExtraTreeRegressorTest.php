<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Regressors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProviderExternal;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\Datasets\Generators\Hyperplane\Hyperplane;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Regressors\ExtraTreeRegressor;
use Rubix\ML\Tests\DataProvider\ExtraTreeRegressorProvider;
use Rubix\ML\Transformers\IntervalDiscretizer;

#[Group('Regressors')]
#[CoversClass(ExtraTreeRegressor::class)]
class ExtraTreeRegressorTest extends TestCase
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
    protected const float MIN_SCORE = 0.89;

    /**
     * Constant used to see the random number generator.
     */
    protected const int RANDOM_SEED = 0;

    protected Hyperplane $generator;

    protected ExtraTreeRegressor $estimator;

    protected RSquared $metric;

    protected function setUp() : void
    {
        $this->generator = new Hyperplane(
            coefficients: [1.0, 5.5, -7, 0.01],
            intercept: 35.0,
            noise: 1.0
        );

        $this->estimator = new ExtraTreeRegressor(
            maxHeight: 30,
            maxLeafSize: 3,
            minPurityIncrease: 1e-7,
            maxFeatures: 4
        );

        $this->metric = new RSquared();

        srand(self::RANDOM_SEED);
    }

    #[Test]
    #[TestDox('Is not trained before training')]
    public function preConditions() : void
    {
        self::assertFalse($this->estimator->trained());
    }

    #[Test]
    #[TestDox('Throws when max height is invalid')]
    public function badMaxDepth() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new ExtraTreeRegressor(0);
    }

    #[Test]
    #[TestDox('Returns estimator type')]
    public function type() : void
    {
        self::assertEquals(EstimatorType::regressor(), $this->estimator->type());
    }

    #[Test]
    #[TestDox('Declares feature compatibility')]
    public function compatibility() : void
    {
        $expected = [
            DataType::categorical(),
            DataType::continuous(),
        ];

        self::assertEquals($expected, $this->estimator->compatibility());
    }

    #[Test]
    #[TestDox('Returns hyperparameters')]
    public function params() : void
    {
        $expected = [
            'max height' => 30,
            'max leaf size' => 3,
            'min purity increase' => 1.0E-7,
            'max features' => 4,
        ];

        self::assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    #[TestDox('Trains, predicts, and returns importances for continuous targets')]
    public function trainPredictImportancesContinuous() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        self::assertTrue($this->estimator->trained());

        $importances = $this->estimator->featureImportances();

        self::assertCount(4, $importances);
        self::assertContainsOnlyFloat($importances);

        $predictions = $this->estimator->predict($testing);

        /** @var list<float|int> $labels */
        $labels = $testing->labels();

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        self::assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    #[TestDox('Can train and predict from provider samples')]
    #[DataProviderExternal(ExtraTreeRegressorProvider::class, 'trainPredictProvider')]
    public function trainPredictAdditional(array $samples, array $labels, array $prediction) : void
    {
        $training = Labeled::quick($samples, $labels);

        $this->estimator->train($training);

        self::assertTrue($this->estimator->trained());

        $importances = $this->estimator->featureImportances();

        self::assertCount(count($samples[0]), $importances);
        self::assertContainsOnlyFloat($importances);

        $predictions = $this->estimator->predict(Unlabeled::quick([$prediction]));

        self::assertIsFloat($predictions[0]);
    }

    #[Test]
    #[TestDox('Trains and predicts with discretized targets')]
    public function trainPredictCategorical() : void
    {
        $training = $this->generator
            ->generate(self::TRAIN_SIZE + self::TEST_SIZE)
            ->apply(new IntervalDiscretizer(bins: 5));

        $testing = $training->randomize()->take(self::TEST_SIZE);

        $this->estimator->train($training);

        self::assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict($testing);

        /** @var list<float|int> $labels */
        $labels = $testing->labels();

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        self::assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    #[TestDox('Throws when predicting before training')]
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

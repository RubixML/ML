<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Regressors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProviderExternal;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\CrossValidation\Metrics\RMSE;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\Datasets\Generators\SwissRoll\SwissRoll;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Regressors\GradientBoost;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Regressors\Ridge;
use Rubix\ML\Tests\DataProvider\GradientBoostProvider;

#[Group('Regressors')]
#[CoversClass(GradientBoost::class)]
class GradientBoostTest extends TestCase
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

    protected SwissRoll $generator;

    protected GradientBoost $estimator;

    protected RSquared $metric;

    protected function setUp() : void
    {
        $this->generator = new SwissRoll(
            x: 4.0,
            y: -7.0,
            z: 0.0,
            scale: 1.0,
            depth: 21.0,
            noise: 0.5
        );

        $this->estimator = new GradientBoost(
            booster: new RegressionTree(maxHeight: 3),
            rate: 0.1,
            ratio: 0.3,
            epochs: 300,
            minChange: 1e-4,
            evalInterval: 3,
            window: 10,
            holdOut: 0.1,
            metric: new RMSE()
        );

        $this->metric = new RSquared();

        srand(self::RANDOM_SEED);
    }

    protected function assertPreConditions() : void
    {
        self::assertFalse($this->estimator->trained());
    }

    #[Test]
    #[TestDox('Throws when booster is incompatible')]
    public function incompatibleBooster() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new GradientBoost(booster: new Ridge());
    }

    #[Test]
    #[TestDox('Throws when learning rate is invalid')]
    public function badLearningRate() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new GradientBoost(booster: null, rate: -1e-3);
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
            'booster' => new RegressionTree(maxHeight: 3),
            'rate' => 0.1,
            'ratio' => 0.3,
            'epochs' => 300,
            'min change' => 0.0001,
            'eval interval' => 3,
            'window' => 10,
            'hold out' => 0.1,
            'metric' => new RMSE(),
        ];

        self::assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    #[TestDox('Trains, predicts, and returns importances')]
    public function trainPredictImportances() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        self::assertTrue($this->estimator->trained());

        $losses = $this->estimator->losses();

        self::assertIsArray($losses);
        self::assertContainsOnlyFloat($losses);

        $scores = $this->estimator->scores();

        self::assertIsArray($scores);
        self::assertContainsOnlyFloat($scores);

        $importances = $this->estimator->featureImportances();

        self::assertCount(3, $importances);
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
    #[TestDox('Returns additional training artifacts and prediction details')]
    #[DataProviderExternal(GradientBoostProvider::class, 'trainPredictAdditionalProvider')]
    public function trainPredictAdditionalChecks(int $trainSize, int $testSize) : void
    {
        $this->estimator->setLogger(new BlackHole());

        $training = $this->generator->generate($trainSize);
        $testing = $this->generator->generate($testSize);

        $this->estimator->train($training);

        self::assertSame(3, $training->numFeatures());

        $losses = $this->estimator->losses();

        self::assertIsArray($losses);
        self::assertNotEmpty($losses);
        self::assertContainsOnlyFloat($losses);

        $scores = $this->estimator->scores();

        self::assertIsArray($scores);
        self::assertNotEmpty($scores);
        self::assertContainsOnlyFloat($scores);

        $importances = $this->estimator->featureImportances();

        self::assertCount(3, $importances);
        self::assertContainsOnlyFloat($importances);
        self::assertGreaterThan(0.0, array_sum($importances));

        $predictions = $this->estimator->predict($testing);

        self::assertCount($testSize, $predictions);
        self::assertContainsOnlyFloat($predictions);
    }

    #[Test]
    #[TestDox('Throws when predicting before training')]
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

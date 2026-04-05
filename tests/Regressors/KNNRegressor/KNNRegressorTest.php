<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Regressors\KNNRegressor;

use Generator;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\DataType;
use Rubix\ML\Datasets\Generators\HalfMoon;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\EstimatorType;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Kernels\Distance\Minkowski;
use Rubix\ML\Regressors\KNNRegressor\KNNRegressor;

#[Group('Regressors')]
#[CoversClass(KNNRegressor::class)]
class KNNRegressorTest extends TestCase
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

    protected KNNRegressor $estimator;

    protected RSquared $metric;

    public static function trainedStateCases() : Generator
    {
        yield 'three-fold partial fit' => [self::TRAIN_SIZE, 3];
    }

    protected function setUp() : void
    {
        $this->generator = new HalfMoon(x: 4.0, y: -7.0, scale: 1.0, rotation: 90, noise: 0.25);

        $this->estimator = new KNNRegressor(k: 10, weighted: true, kernel:  new Minkowski(3.0));

        $this->metric = new RSquared();

        srand(self::RANDOM_SEED);
    }

    #[Test]
    #[TestDox('asserts preconditions')]
    public function assertsPreConditions() : void
    {
        self::assertFalse($this->estimator->trained());
    }

    #[Test]
    #[TestDox('rejects invalid k values')]
    public function rejectsInvalidK() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new KNNRegressor(k: 0);
    }

    #[Test]
    #[TestDox('returns the regressor estimator type')]
    public function returnsTheRegressorEstimatorType() : void
    {
        self::assertEquals(EstimatorType::regressor(), $this->estimator->type());
    }

    #[Test]
    #[TestDox('returns the expected compatibility types')]
    public function returnsTheExpectedCompatibilityTypes() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        self::assertEquals($expected, $this->estimator->compatibility());
    }

    #[Test]
    #[TestDox('returns the configured parameters')]
    public function returnsTheConfiguredParameters() : void
    {
        $expected = [
            'k' => 10,
            'weighted' => true,
            'kernel' => new Minkowski(3.0),
        ];

        self::assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    #[TestDox('trains partially and makes accurate predictions')]
    public function trainsPartiallyAndMakesAccuratePredictions() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $folds = $training->fold(3);

        $this->estimator->train($folds[0]);
        $this->estimator->partial($folds[1]);
        $this->estimator->partial($folds[2]);

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
    #[TestDox('rejects incompatible training data')]
    public function rejectsIncompatibleTrainingData() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick(samples: [['bad']], labels: [2]));
    }

    #[Test]
    #[TestDox('rejects predictions from an untrained model')]
    public function rejectsPredictionsFromAnUntrainedModel() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }

    #[Test]
    #[TestDox('becomes trained after partial fitting')]
    #[DataProvider('trainedStateCases')]
    public function becomesTrainedAfterPartialFitting(int $trainSize, int $folds) : void
    {
        $training = $this->generator->generate($trainSize);

        $parts = $training->fold($folds);

        $this->estimator->train($parts[0]);

        for ($i = 1; $i < $folds; ++$i) {
            $this->estimator->partial($parts[$i]);
        }

        self::assertTrue($this->estimator->trained());
    }
}

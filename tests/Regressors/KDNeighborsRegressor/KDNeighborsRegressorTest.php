<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Regressors\KDNeighborsRegressor;

use PHPUnit\Framework\Attributes\CoversClass;
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
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Regressors\KDNeighborsRegressor\KDNeighborsRegressor;

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
    protected const float MIN_SCORE = 0.9;

    /**
     * Constant used to see the random number generator.
     */
    protected const int RANDOM_SEED = 0;

    protected HalfMoon $generator;

    protected KDNeighborsRegressor $estimator;

    protected RSquared $metric;

    protected function setUp() : void
    {
        $this->generator = new HalfMoon(x: 4.0, y: -7.0, scale: 1.0, rotation: 90, noise: 0.25);

        $this->estimator = new KDNeighborsRegressor(k: 5, weighted: true, tree: new KDTree());

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

        new KDNeighborsRegressor(k: 0);
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
            'k' => 5,
            'weighted' => true,
            'tree' => new KDTree(),
        ];

        self::assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    #[TestDox('trains and makes accurate predictions')]
    public function trainsAndMakesAccuratePredictions() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        self::assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict($testing);

        self::assertCount($testing->numSamples(), $predictions);

        foreach ($predictions as $prediction) {
            self::assertIsFloat($prediction);
            self::assertFalse(is_nan($prediction));
        }

        /** @var list<int|float> $labels */
        $labels = $testing->labels();

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        self::assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    #[TestDox('serialization preserves the trained model and predictions')]
    public function serializationPreservesTheTrainedModelAndPredictions() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $predictionsBefore = $this->estimator->predict($testing);

        $copy = unserialize(serialize($this->estimator));

        self::assertInstanceOf(KDNeighborsRegressor::class, $copy);
        self::assertTrue($copy->trained());
        self::assertInstanceOf(KDTree::class, $copy->tree());

        $predictionsAfter = $copy->predict($testing);

        self::assertCount($testing->numSamples(), $predictionsAfter);

        foreach ($predictionsAfter as $i => $prediction) {
            self::assertIsFloat($prediction);
            self::assertFalse(is_nan($prediction));
            self::assertEqualsWithDelta((float) $predictionsBefore[$i], (float) $prediction, 1e-8);
        }
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
}

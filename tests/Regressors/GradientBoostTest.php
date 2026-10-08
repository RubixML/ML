<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Regressors;

use Generator;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Tuple;
use Rubix\ML\CrossValidation\Metrics\RMSE;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\Datasets\Generators\SwissRoll;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Exceptions\EmptyDataset;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\IncorrectDatasetDimensionality;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Regressors\GradientBoost;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Regressors\Ridge;

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

    /**
     * @return Generator<string, array{0: int, 1: int}>
     */
    public static function trainPredictAdditionalProvider() : Generator
    {
        yield 'default swiss roll sample' => [512, 256];

        yield 'smaller swiss roll sample' => [128, 64];
    }

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
    public function windowDisabled() : void
    {
        srand(self::RANDOM_SEED);

        $estimator = new GradientBoost(
            booster: new RegressionTree(maxHeight: 3),
            rate: 0.1,
            ratio: 0.3,
            epochs: 10,
            minChange: 1e-12,
            evalInterval: 1,
            window: 0,
            metric: new RMSE()
        );

        $estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TEST_SIZE);

        $estimator->train($training);

        self::assertTrue($estimator->trained());
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
            'metric' => new RMSE(),
        ];

        self::assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    #[TestDox('Assert the iterative progress contract')]
    public function progressContract() : void
    {
        self::assertSame([], iterator_to_array($this->estimator->progress(), false));

        srand(self::RANDOM_SEED);

        $estimator = new GradientBoost(
            booster: new RegressionTree(maxHeight: 3),
            rate: 0.1,
            ratio: 0.3,
            epochs: 5,
            minChange: 1e-6,
            evalInterval: 1,
            metric: new RMSE()
        );

        $estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $estimator->train($training);

        self::assertTrue($estimator->trained());

        $losses = $estimator->losses();

        self::assertIsArray($losses);
        self::assertNotEmpty($losses);

        $rows = iterator_to_array($estimator->progress(), false);

        self::assertCount(count($losses), $rows);

        foreach ($rows as $row) {
            self::assertIsInt($row['Epoch']);
            self::assertIsFloat($row['L2 Loss']);
            self::assertArrayHasKey('RMSE', $row);
        }

        self::assertSame(array_values($losses), array_column($rows, 'L2 Loss'));
        self::assertSame(array_keys($losses), array_column($rows, 'Epoch'));
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
    #[DataProvider('trainPredictAdditionalProvider')]
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
        self::assertEmpty($scores);

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

    #[Test]
    public function restoreStateFromSerializedModel() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $restored = unserialize(serialize($this->estimator));

        $this->assertTrue($restored->trained());

        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->assertEquals($this->estimator->predict($testing), $restored->predict($testing));
    }

    #[Test]
    public function earlyStoppingRestoresBestScoringEnsembleState() : void
    {
        [$validation, $training] = $this->generator->generate(self::TRAIN_SIZE)->randomize()->split(0.2);

        $estimator = new GradientBoost(
            booster: new RegressionTree(maxHeight: 3),
            rate: 0.5,
            ratio: 0.5,
            epochs: 5,
            minChange: 0.0,
            evalInterval: 1,
            window: 0,
            metric: new ScriptedMetric([0.5, 0.9, 0.7, 0.6, 0.55])
        );

        $estimator->setValidationDataset($validation);

        $estimator->train($training);

        $scores = $estimator->scores();

        $this->assertIsArray($scores);
        $this->assertSame([1, 2, 3, 4, 5], array_keys($scores));

        $bestEpoch = array_search(max($scores), $scores);

        $this->assertSame(2, $bestEpoch);

        $ensemble = $estimator->__serialize()['ensemble'];

        $this->assertCount($bestEpoch - 1, $ensemble);
    }

    #[Test]
    #[TestDox('Injected validation dataset enables progress monitoring and early stopping')]
    public function injectedValidationDatasetEnablesScoring() : void
    {
        srand(self::RANDOM_SEED);

        $dataset = $this->generator->generate(self::TRAIN_SIZE + self::TEST_SIZE);

        [$testing, $training] = $dataset->randomize()->split(0.5);

        $estimator = $this->buildEstimator();

        $estimator->train($training);

        self::assertTrue($estimator->trained());
        self::assertEmpty($estimator->scores());

        $estimator = $this->buildEstimator();

        $estimator->setValidationDataset($testing);

        $estimator->train($training);

        self::assertTrue($estimator->trained());
        self::assertNotEmpty($estimator->scores());
    }

    #[Test]
    #[TestDox('Injected validation dataset is not persisted')]
    public function injectedValidationDatasetIsNotPersisted() : void
    {
        srand(self::RANDOM_SEED);

        $dataset = $this->generator->generate(self::TRAIN_SIZE + self::TEST_SIZE);

        [$testing, $training] = $dataset->randomize()->split(0.5);

        $estimator = $this->buildEstimator();

        $estimator->setValidationDataset($testing);

        $estimator->train($training);

        $serialized = $estimator->__serialize();

        self::assertArrayNotHasKey('validationDataset', $serialized);

        $copy = unserialize(serialize($estimator));

        self::assertTrue($copy->trained());
        self::assertArrayNotHasKey('validationDataset', $copy->__serialize());
    }

    #[Test]
    #[TestDox('Null injected validation dataset disables progress monitoring and early stopping')]
    public function nullInjectedValidationDatasetDisablesScoring() : void
    {
        srand(self::RANDOM_SEED);

        $dataset = $this->generator->generate(self::TRAIN_SIZE + self::TEST_SIZE);

        [$testing, $training] = $dataset->randomize()->split(0.5);

        $estimator = $this->buildEstimator();

        $estimator->setValidationDataset($testing);
        $estimator->setValidationDataset(null);

        $estimator->train($training);

        self::assertTrue($estimator->trained());
        self::assertEmpty($estimator->scores());
    }

    #[Test]
    #[TestDox('Injected validation dataset must not be empty')]
    public function injectedValidationDatasetRejectsEmptyDataset() : void
    {
        $this->expectException(EmptyDataset::class);

        $this->estimator->setValidationDataset(Labeled::quick());
    }

    #[Test]
    #[TestDox('Injected validation dataset must match the training dimensionality')]
    public function injectedValidationDatasetRejectsMismatchedDimensionality() : void
    {
        srand(self::RANDOM_SEED);

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $validation = Labeled::quick(
            samples: [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            labels: [1.0, 2.0]
        );

        $this->estimator->setValidationDataset($validation);

        $this->expectException(IncorrectDatasetDimensionality::class);

        $this->estimator->train($training);
    }

    /**
     * Build an estimator with a small epoch budget.
     *
     * @return GradientBoost
     */
    private function buildEstimator() : GradientBoost
    {
        return new GradientBoost(
            booster: new RegressionTree(maxHeight: 3),
            rate: 0.1,
            ratio: 0.3,
            epochs: 10,
            minChange: 0.0,
            evalInterval: 1,
            window: 0,
            metric: new RMSE()
        );
    }
}

class ScriptedMetric implements Metric
{
    /**
     * @var float[]
     */
    private array $scores;

    /**
     * @param float[] $scores
     */
    public function __construct(array $scores)
    {
        $this->scores = $scores;
    }

    public function range() : Tuple
    {
        return new Tuple(-INF, 1.0);
    }

    public function compatibility() : array
    {
        return [EstimatorType::regressor()];
    }

    public function score(array $predictions, array $labels) : float
    {
        return array_shift($this->scores) ?? 0.0;
    }

    public function __toString() : string
    {
        return 'Scripted Metric';
    }
}

<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Classifiers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\Tuple;
use Rubix\ML\EstimatorType;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\CrossValidation\Metrics\Metric;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\LogitBoost;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Exceptions\EmptyDataset;
use Rubix\ML\Exceptions\IncorrectDatasetDimensionality;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Classifiers')]
#[CoversClass(LogitBoost::class)]
class LogitBoostTest extends TestCase
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

    protected Agglomerate $generator;

    protected LogitBoost $estimator;

    protected FBeta $metric;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'inner' => new Circle(
                    x: 0.0,
                    y: 0.0,
                    scale: 5.0,
                    noise: 0.05
                ),
                'outer' => new Circle(
                    x: 0.0,
                    y: 0.0,
                    scale: 10.0,
                    noise: 0.1
                ),
            ],
            weights: [0.4, 0.6]
        );

        $this->estimator = new LogitBoost(
            booster: new RegressionTree(3),
            rate: 0.1,
            ratio: 0.5,
            epochs: 1000,
            minChange: 1e-4,
            evalInterval: 3,
            window: 5,
            metric: new FBeta()
        );

        $this->metric = new FBeta();

        srand(self::RANDOM_SEED);
    }

    #[Test]
    public function preConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    #[Test]
    public function windowDisabled() : void
    {
        srand(self::RANDOM_SEED);

        $estimator = new LogitBoost(
            booster: new RegressionTree(3),
            rate: 0.1,
            ratio: 0.5,
            epochs: 10,
            minChange: 1e-12,
            evalInterval: 1,
            window: 0,
            metric: new FBeta()
        );

        $estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TEST_SIZE);

        $estimator->train($training);

        $this->assertTrue($estimator->trained());
    }

    #[Test]
    public function progressContract() : void
    {
        $this->assertSame([], iterator_to_array($this->estimator->progress(), false));

        srand(self::RANDOM_SEED);

        $estimator = new LogitBoost(
            booster: new RegressionTree(3),
            rate: 0.1,
            ratio: 0.5,
            epochs: 5,
            minChange: 1e-6,
            evalInterval: 1,
            metric: new FBeta()
        );

        $estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $estimator->train($training);

        $this->assertTrue($estimator->trained());

        $losses = $estimator->losses();

        $this->assertIsArray($losses);
        $this->assertNotEmpty($losses);

        $rows = iterator_to_array($estimator->progress(), false);

        $this->assertCount(count($losses), $rows);

        foreach ($rows as $row) {
            $this->assertIsInt($row['Epoch']);
            $this->assertIsFloat($row['Cross Entropy']);
            $this->assertArrayHasKey('F Beta (beta: 1)', $row);
        }

        $this->assertSame(array_values($losses), array_column($rows, 'Cross Entropy'));
        $this->assertSame(array_keys($losses), array_column($rows, 'Epoch'));
    }

    #[Test]
    public function type() : void
    {
        $this->assertEquals(EstimatorType::classifier(), $this->estimator->type());
    }

    #[Test]
    public function compatibility() : void
    {
        $expected = [
            DataType::categorical(),
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    #[Test]
    public function params() : void
    {
        $expected = [
            'booster' => new RegressionTree(3),
            'rate' => 0.1,
            'ratio' => 0.5,
            'epochs' => 1000,
            'min change' => 0.0001,
            'eval interval' => 3,
            'window' => 5,
            'metric' => new FBeta(1),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    public function trainPredict() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $scores = $this->estimator->losses();

        $this->assertIsArray($scores);
        $this->assertContainsOnlyFloat($scores);

        $losses = $this->estimator->losses();

        $this->assertIsArray($losses);
        $this->assertContainsOnlyFloat($losses);

        $importances = $this->estimator->featureImportances();

        $this->assertIsArray($importances);
        $this->assertCount(2, $importances);
        $this->assertContainsOnlyFloat($importances);

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $testing->labels()
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }

    #[Test]
    public function restoreStateFromSerializedModel() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $restored = unserialize(serialize($this->estimator));

        $this->assertTrue($restored->trained());

        $this->assertEquals($this->estimator->predict($testing), $restored->predict($testing));
    }

    #[Test]
    public function probaRowsSumToOne() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $probabilities = $this->estimator->proba($testing);

        $this->assertIsArray($probabilities);
        $this->assertCount(self::TEST_SIZE, $probabilities);

        foreach ($probabilities as $probability) {
            $this->assertEqualsWithDelta(1.0, array_sum($probability), 1e-8);
        }
    }

    #[Test]
    public function earlyStoppingRestoresBestScoringEnsembleState() : void
    {
        [$validation, $training] = $this->generator->generate(self::TRAIN_SIZE)->randomize()->split(0.2);

        $estimator = new LogitBoost(
            booster: new RegressionTree(3),
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

        $boosters = $estimator->__serialize()['boosters'];

        $this->assertCount($bestEpoch - 1, $boosters);
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

        $this->assertTrue($estimator->trained());
        $this->assertEmpty($estimator->scores());

        $estimator = $this->buildEstimator();

        $estimator->setValidationDataset($testing);

        $estimator->train($training);

        $this->assertTrue($estimator->trained());
        $this->assertNotEmpty($estimator->scores());
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

        $this->assertArrayNotHasKey('validationDataset', $serialized);

        $copy = unserialize(serialize($estimator));

        $this->assertTrue($copy->trained());
        $this->assertArrayNotHasKey('validationDataset', $copy->__serialize());
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

        $this->assertTrue($estimator->trained());
        $this->assertEmpty($estimator->scores());
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
            samples: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            labels: ['inner', 'outer']
        );

        $this->estimator->setValidationDataset($validation);

        $this->expectException(IncorrectDatasetDimensionality::class);

        $this->estimator->train($training);
    }

    #[Test]
    #[TestDox('Injected validation dataset rejects labels unknown to the classifier')]
    public function injectedValidationDatasetRejectsUnknownLabels() : void
    {
        srand(self::RANDOM_SEED);

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $validation = Labeled::quick(
            samples: [[1.0, 2.0], [3.0, 4.0]],
            labels: ['purple', 'purple']
        );

        $this->estimator->setValidationDataset($validation);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessageMatches('/unknown to this classifier/');

        $this->estimator->train($training);
    }

    /**
     * Build an estimator with a small epoch budget.
     *
     * @return LogitBoost
     */
    private function buildEstimator() : LogitBoost
    {
        return new LogitBoost(
            booster: new RegressionTree(3),
            rate: 0.5,
            ratio: 0.5,
            epochs: 10,
            minChange: 0.0,
            evalInterval: 1,
            window: 0,
            metric: new FBeta()
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
        return new Tuple(0.0, 1.0);
    }

    public function compatibility() : array
    {
        return [EstimatorType::classifier()];
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

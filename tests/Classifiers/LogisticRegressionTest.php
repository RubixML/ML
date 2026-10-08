<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Classifiers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\NeuralNet\CostFunctions\BinaryCrossEntropy;
use Rubix\ML\Exceptions\EmptyDataset;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\IncorrectDatasetDimensionality;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

use function sys_get_temp_dir;
use function uniqid;

#[Group('Classifiers')]
#[CoversClass(LogisticRegression::class)]
class LogisticRegressionTest extends TestCase
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

    protected LogisticRegression $estimator;

    protected FBeta $metric;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'male' => new Blob(
                    center: [69.2, 195.7, 40.0],
                    stdDev: [2.0, 6.0, 0.6]
                ),
                'female' => new Blob(
                    center: [63.7, 168.5, 38.1],
                    stdDev: [1.6, 5.0, 0.8]
                ),
            ],
            weights: [0.45, 0.55]
        );

        $this->estimator = new LogisticRegression(
            batchSize: 100,
            optimizer: new Adam(new Constant(0.01)),
            l2Penalty: 1e-4,
            epochs: 300,
            minChange: 1e-4,
            evalInterval: 3,
            window: 5,
            costFn: new BinaryCrossEntropy(),
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

        $estimator = new LogisticRegression(
            batchSize: 100,
            optimizer: new Adam(new Constant(0.01)),
            l2Penalty: 1e-4,
            epochs: 10,
            minChange: 1e-12,
            evalInterval: 1,
            window: 0,
            costFn: new BinaryCrossEntropy(),
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

        $estimator = new LogisticRegression(
            batchSize: 32,
            optimizer: new Adam(new Constant(0.01)),
            l2Penalty: 1e-4,
            epochs: 5,
            minChange: 1e-6,
            evalInterval: 1,
            costFn: new BinaryCrossEntropy(),
            metric: new FBeta()
        );

        $estimator->setLogger(new BlackHole());

        $dataset = $this->generator->generate(self::TRAIN_SIZE);

        $dataset->apply(new ZScaleStandardizer());

        $estimator->train($dataset);

        $this->assertTrue($estimator->trained());

        $losses = $estimator->losses();

        $this->assertIsArray($losses);
        $this->assertNotEmpty($losses);

        $rows = iterator_to_array($estimator->progress(), false);

        $this->assertCount(count($losses), $rows);

        foreach ($rows as $row) {
            $this->assertIsInt($row['Epoch']);
            $this->assertIsFloat($row['Binary Cross Entropy']);
            $this->assertArrayHasKey('F Beta (beta: 1)', $row);
        }

        $this->assertSame(array_values($losses), array_column($rows, 'Binary Cross Entropy'));
        $this->assertSame(array_keys($losses), array_column($rows, 'Epoch'));
    }

    #[Test]
    public function badBatchSize() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new LogisticRegression(batchSize: -100);
    }

    #[Test]
    public function negativeL1Penalty() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new LogisticRegression(l1Penalty: -1.0);
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
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    #[Test]
    public function params() : void
    {
        $expected = [
            'batch size' => 100,
            'optimizer' => new Adam(new Constant(0.01)),
            'l1 penalty' => 1e-4,
            'l2 penalty' => 1e-4,
            'epochs' => 300,
            'min change' => 1e-4,
            'eval interval' => 3,
            'window' => 5,
            'cost fn' => new BinaryCrossEntropy(),
            'metric' => new FBeta(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    public function trainPartialPredict() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $dataset = $this->generator->generate(self::TRAIN_SIZE + self::TEST_SIZE);

        $dataset->apply(new ZScaleStandardizer());

        $testing = $dataset->randomize()->take(self::TEST_SIZE);

        $folds = $dataset->stratifiedFold(3);

        $this->estimator->train($folds[0]);
        $this->estimator->partial($folds[1]);
        $this->estimator->partial($folds[2]);

        $this->assertTrue($this->estimator->trained());

        $losses = $this->estimator->losses();

        $this->assertIsArray($losses);
        $this->assertContainsOnlyFloat($losses);

        $scores = $this->estimator->scores();

        $this->assertIsArray($scores);
        $this->assertContainsOnlyFloat($scores);

        $importances = $this->estimator->featureImportances();

        $this->assertIsArray($importances);
        $this->assertCount(3, $importances);
        $this->assertContainsOnlyFloat($importances);

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $testing->labels()
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    public function snapshotPathIsTransientAndResolvedLazily() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $dataset = $this->generator->generate(self::TRAIN_SIZE + self::TEST_SIZE);

        $dataset->apply(new ZScaleStandardizer());

        $snapshotPath = sys_get_temp_dir() . '/rubix-ml-test-' . uniqid() . '.dat';

        $this->estimator->setSnapshotPath($snapshotPath);

        $this->estimator->train($dataset->stratifiedFold(2)[0]);

        $this->assertTrue($this->estimator->trained());

        $this->assertArrayNotHasKey('snapshotPath', $this->estimator->__serialize());

        $copy = unserialize(serialize($this->estimator));

        $this->assertTrue($copy->trained());

        $this->assertArrayNotHasKey('snapshotPath', $copy->__serialize());

        $copy->partial($dataset->stratifiedFold(2)[0]);

        $this->assertArrayNotHasKey('snapshotPath', $copy->__serialize());
    }

    #[Test]
    public function snapshotPathRejectsDirectory() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->setSnapshotPath(sys_get_temp_dir());
    }

    #[Test]
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick(samples: [['bad']], labels: ['green']));
    }

    #[Test]
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
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
    #[TestDox('Training does not mutate the labels of the given dataset')]
    public function trainingDoesNotMutateGivenDataset() : void
    {
        srand(self::RANDOM_SEED);

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $labels = $training->labels();

        $estimator = $this->buildEstimator();

        $estimator->train($training);

        $this->assertTrue($estimator->trained());
        $this->assertSame($labels, $training->labels());
    }

    #[Test]
    #[TestDox('Injected validation dataset is retained across partial training')]
    public function injectedValidationDatasetIsRetainedAcrossPartialTraining() : void
    {
        srand(self::RANDOM_SEED);

        $dataset = $this->generator->generate(self::TRAIN_SIZE + self::TEST_SIZE);

        [$testing, $training] = $dataset->randomize()->split(0.5);

        $estimator = $this->buildEstimator();

        $estimator->setValidationDataset($testing);

        $estimator->train($training->fold(2)[0]);

        $this->assertNotEmpty($estimator->scores());

        $estimator->partial($training->fold(2)[1]);

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
            samples: [[1.0, 2.0], [3.0, 4.0]],
            labels: ['male', 'female']
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
            samples: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            labels: ['other', 'other']
        );

        $this->estimator->setValidationDataset($validation);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessageMatches('/unknown to this classifier/');

        $this->estimator->train($training);
    }

    /**
     * Build an estimator with a small epoch budget.
     *
     * @return LogisticRegression
     */
    private function buildEstimator() : LogisticRegression
    {
        return new LogisticRegression(
            batchSize: 100,
            optimizer: new Adam(new Constant(0.01)),
            l2Penalty: 1e-4,
            epochs: 10,
            minChange: 0.0,
            evalInterval: 1,
            window: 0,
            costFn: new BinaryCrossEntropy(),
            metric: new FBeta()
        );
    }
}

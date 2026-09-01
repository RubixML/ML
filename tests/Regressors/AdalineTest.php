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
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\CrossValidation\Metrics\RMSE;
use Rubix\ML\Datasets\Generators\Hyperplane;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Regressors\Adaline;

use function sys_get_temp_dir;
use function uniqid;

#[Group('Regressors')]
#[CoversClass(Adaline::class)]
class AdalineTest extends TestCase
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

    protected Hyperplane $generator;

    protected Adaline $estimator;

    protected RSquared $metric;

    /**
     * @return Generator<string, array{0: list<list<int>>, 1: list<int>, 2: list<int>}>
     */
    public static function trainPredictProvider() : Generator
    {
        yield '1 feature linear sample' => [
            [
                [0.0],
                [1.0],
                [2.0],
                [3.0],
            ],
            [3.0, 5.0, 7.0, 9.0],
            [4.0],
        ];

        yield '2 feature linear sample' => [
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [1.0, 2.0],
            ],
            [3.0, 6.0, 7.0, 8.0],
            [2.0, 2.0],
        ];

        yield '3 feature linear sample' => [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [4.0, 5.0, 6.0, 7.0],
            [1.0, 1.0, 1.0],
        ];
    }

    protected function setUp() : void
    {
        $this->generator = new Hyperplane(
            coefficients: [1.0, 5.5, -7, 0.01],
            intercept: 0.0,
            noise: 1.0
        );

        $this->estimator = new Adaline(
            batchSize: 32,
            optimizer: new Adam(rate: 0.001),
            l2Penalty: 1e-4,
            epochs: 100,
            minChange: 1e-4,
            evalInterval: 3,
            window: 5,
            holdOut: 0.1,
            costFn: new HuberLoss(1.0),
            metric: new RMSE()
        );

        $this->metric = new RSquared();

        srand(self::RANDOM_SEED);
    }

    #[Test]
    #[TestDox('Assert pre conditions')]
    public function preConditions() : void
    {
        self::assertFalse($this->estimator->trained());
    }

    #[Test]
    #[TestDox('Throws an exception for a bad batch size')]
    public function badBatchSize() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Adaline(-100);
    }

    #[Test]
    #[TestDox('Reports the estimator type')]
    public function type() : void
    {
        self::assertEquals(EstimatorType::regressor(), $this->estimator->type());
    }

    #[Test]
    #[TestDox('Reports compatibility')]
    public function compatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        self::assertEquals($expected, $this->estimator->compatibility());
    }

    #[Test]
    #[TestDox('Reports parameters')]
    public function params() : void
    {
        $expected = [
            'batch size' => 32,
            'optimizer' => new Adam(0.001),
            'l2 penalty' => 1e-4,
            'epochs' => 100,
            'min change' => 1e-4,
            'eval interval' => 3,
            'window' => 5,
            'hold out' => 0.1,
            'cost fn' => new HuberLoss(1.0),
            'metric' => new RMSE(),
        ];

        self::assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    #[TestDox('Can train, predict, and provide feature importances')]
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

        self::assertCount(4, $importances);
        self::assertContainsOnlyFloat($importances);

        $predictions = $this->estimator->predict($testing);

        /** @var list<float> $labels */
        $labels = $testing->labels();
        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        self::assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    #[TestDox('Casts the data type of every neural network NDArray in place')]
    public function setDataType() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(128);

        $this->estimator->train($training);

        self::assertTrue($this->estimator->trained());

        $this->estimator->setDataType('float32');

        self::assertSame('float32', $this->estimator->dataType());

        $network = $this->estimator->network();

        self::assertNotNull($network);

        foreach ($network->layers() as $layer) {
            if ($layer instanceof Parametric) {
                foreach ($layer->parameters() as $parameter) {
                    self::assertSame('float32', $parameter->param()->dataType());
                }
            }
        }
    }

    #[Test]
    #[TestDox('Throws when the given data type is not float32')]
    public function setDataTypeInvalid() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->setDataType('float64');
    }

    #[Test]
    #[TestDox('Trains the network with the configured data type')]
    public function setDataTypeBeforeTraining() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $this->estimator->setDataType('float32');

        $training = $this->generator->generate(128);

        $this->estimator->train($training);

        $network = $this->estimator->network();

        self::assertNotNull($network);

        foreach ($network->layers() as $layer) {
            if ($layer instanceof Parametric) {
                foreach ($layer->parameters() as $parameter) {
                    self::assertSame('float32', $parameter->param()->dataType());
                }
            }
        }
    }

    #[Test]
    #[TestDox('Snapshot path is transient and resolved lazily')]
    public function snapshotPathIsTransientAndResolvedLazily() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $dataset = $this->generator->generate(self::TRAIN_SIZE + self::TEST_SIZE);

        $snapshotPath = sys_get_temp_dir() . '/rubix-ml-test-' . uniqid() . '.dat';

        $this->estimator->setSnapshotPath($snapshotPath);

        $this->estimator->train($dataset->fold(2)[0]);

        self::assertTrue($this->estimator->trained());

        self::assertArrayNotHasKey('snapshotPath', $this->estimator->__serialize());

        $copy = unserialize(serialize($this->estimator));

        self::assertTrue($copy->trained());

        self::assertArrayNotHasKey('snapshotPath', $copy->__serialize());

        $copy->partial($dataset->fold(2)[0]);

        self::assertArrayNotHasKey('snapshotPath', $copy->__serialize());
    }

    #[Test]
    #[TestDox('Snapshot path rejects a directory')]
    public function snapshotPathRejectsDirectory() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->setSnapshotPath(sys_get_temp_dir());
    }

    #[Test]
    #[TestDox('Throws an exception when training with incompatible data')]
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick(samples: [['bad']], labels: [2]));
    }

    #[Test]
    #[TestDox('Throws an exception when predicting before training')]
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }

    #[Test]
    #[TestDox('Trains, predicts, and returns acceptable Adaline values')]
    #[DataProvider('trainPredictProvider')]
    public function trainPredict(array $samples, array $labels, array $prediction) : void
    {
        $estimator = new Adaline(
            batchSize: 32,
            optimizer: new Adam(rate: 0.001),
            l2Penalty: 1e-4,
            epochs: 100,
            minChange: 1e-4,
            costFn: new HuberLoss(1.0)
        );

        $training = Labeled::quick($samples, $labels);
        $estimator->train($training);

        self::assertTrue($estimator->trained());
        $params = $estimator->params();

        self::assertSame(32, $params['batch size']);
        self::assertEquals(1e-4, $params['l2 penalty']);
        self::assertSame(100, $params['epochs']);
        self::assertEquals(1e-4, $params['min change']);

        $predictions = $estimator->predict(Unlabeled::quick([$prediction]));

        self::assertIsFloat($predictions[0]);
    }
}

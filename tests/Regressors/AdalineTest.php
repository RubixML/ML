<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Regressors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProviderExternal;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\Datasets\Generators\Hyperplane;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss\HuberLoss;
use Rubix\ML\NeuralNet\Optimizers\Adam\Adam;
use Rubix\ML\Regressors\Adaline;
use Rubix\ML\Tests\DataProvider\AdalineProvider;

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
            window: 5,
            costFn: new HuberLoss(1.0)
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
            'window' => 5,
            'cost fn' => new HuberLoss(1.0),
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
    #[DataProviderExternal(AdalineProvider::class, 'trainPredictProvider')]
    public function trainPredict(array $samples, array $labels, array $prediction) : void
    {
        $estimator = new Adaline(
            batchSize: 32,
            optimizer: new Adam(rate: 0.001),
            l2Penalty: 1e-4,
            epochs: 100,
            minChange: 1e-4,
            window: 5,
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
        self::assertSame(5, $params['window']);

        $predictions = $estimator->predict(Unlabeled::quick([$prediction]));

        self::assertIsFloat($predictions[0]);
    }
}

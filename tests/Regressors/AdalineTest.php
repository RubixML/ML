<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Regressors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Regressors\Adaline;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Datasets\Generators\Hyperplane;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

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

    public function testAssertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    public function testBadBatchSize() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Adaline(-100);
    }

    public function testType() : void
    {
        $this->assertEquals(EstimatorType::regressor(), $this->estimator->type());
    }

    public function testCompatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    public function testParams() : void
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

        $this->assertEquals($expected, $this->estimator->params());
    }

    public function testTrainPredictImportances() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $losses = $this->estimator->losses();

        $this->assertIsArray($losses);
        $this->assertContainsOnlyFloat($losses);

        $importances = $this->estimator->featureImportances();

        $this->assertCount(4, $importances);
        $this->assertContainsOnlyFloat($importances);

        $predictions = $this->estimator->predict($testing);

        /** @var list<float> $labels */
        $labels = $testing->labels();
        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    public function testTrainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick(samples: [['bad']], labels: [2]));
    }

    public function testPredictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

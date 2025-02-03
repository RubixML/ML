<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Regressors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Regressors\Ridge;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Hyperplane;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Regressors')]
#[CoversClass(Ridge::class)]
class RidgeTest extends TestCase
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

    protected Ridge $estimator;

    protected RSquared $metric;

    protected function setUp() : void
    {
        $this->generator = new Hyperplane(
            coefficients: [1.0, 5.5, -7, 0.01],
            intercept: 0.0,
            noise: 1.0
        );

        $this->estimator = new Ridge(1.0);

        $this->metric = new RSquared();

        srand(self::RANDOM_SEED);
    }

    public function testAssertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    public function testBadL2Penalty() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Ridge(-1e-4);
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

    public function testTrainPredictImportances() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $coefficients = $this->estimator->coefficients();

        $this->assertIsArray($coefficients);
        $this->assertCount(4, $coefficients);

        $this->assertIsFloat($this->estimator->bias());

        $importances = $this->estimator->featureImportances();

        $this->assertCount(4, $importances);
        $this->assertContainsOnlyFloat($importances);

        $predictions = $this->estimator->predict($testing);

        /** @var list<int|float> $labels */
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

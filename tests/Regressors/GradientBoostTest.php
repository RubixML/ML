<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Regressors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Regressors\Ridge;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Regressors\GradientBoost;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\CrossValidation\Metrics\RMSE;
use Rubix\ML\Datasets\Generators\SwissRoll;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

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
        $this->assertFalse($this->estimator->trained());
    }

    public function testIncompatibleBooster() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new GradientBoost(booster: new Ridge());
    }

    public function testBadLearningRate() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new GradientBoost(booster: null, rate: -1e-3);
    }

    public function testType() : void
    {
        $this->assertEquals(EstimatorType::regressor(), $this->estimator->type());
    }

    public function testCompatibility() : void
    {
        $expected = [
            DataType::categorical(),
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    public function testParams() : void
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

        $scores = $this->estimator->scores();

        $this->assertIsArray($scores);
        $this->assertContainsOnlyFloat($scores);

        $importances = $this->estimator->featureImportances();

        $this->assertCount(3, $importances);
        $this->assertContainsOnlyFloat($importances);

        $predictions = $this->estimator->predict($testing);

        /** @var list<float|int> $labels */
        $labels = $testing->labels();

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    public function testPredictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

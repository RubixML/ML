<?php

namespace Rubix\ML\Tests\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Other\Loggers\BlackHole;
use Rubix\ML\Regressors\GradientBoost;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Regressors\DummyRegressor;
use Rubix\ML\CrossValidation\Metrics\RMSE;
use Rubix\ML\Datasets\Generators\SwissRoll;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

/**
 * @group Regressors
 * @covers \Rubix\ML\Regressors\GradientBoost
 */
class GradientBoostTest extends TestCase
{
    protected const TRAIN_SIZE = 400;
    protected const TEST_SIZE = 10;
    protected const MIN_SCORE = 0.9;

    protected const RANDOM_SEED = 0;

    /**
     * @var \Rubix\ML\Datasets\Generators\SwissRoll
     */
    protected $generator;

    /**
     * @var \Rubix\ML\Regressors\GradientBoost
     */
    protected $estimator;

    /**
     * @var \Rubix\ML\CrossValidation\Metrics\RSquared
     */
    protected $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new SwissRoll(4.0, -7.0, 0.0, 1.0, 0.3);

        $this->estimator = new GradientBoost(new RegressionTree(3), 0.3, 0.3, 300, 1e-4, 10, 0.1, new RMSE());
    
        $this->metric = new RSquared();

        srand(self::RANDOM_SEED);
    }

    protected function assertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }
    
    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(GradientBoost::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    /**
     * @test
     */
    public function incompatibleBooster() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new GradientBoost(new DummyRegressor());
    }

    /**
     * @test
     */
    public function badLearningRate() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new GradientBoost(null, -1e-3);
    }

    /**
     * @test
     */
    public function type() : void
    {
        $this->assertEquals(EstimatorType::regressor(), $this->estimator->type());
    }

    /**
     * @test
     */
    public function compatibility() : void
    {
        $expected = [
            DataType::categorical(),
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    /**
     * @test
     */
    public function params() : void
    {
        $expected = [
            'booster' => new RegressionTree(3),
            'rate' => 0.3,
            'ratio' => 0.3,
            'estimators' => 300,
            'min_change' => 0.0001,
            'window' => 10,
            'hold_out' => 0.1,
            'metric' => new RMSE(),
            'base' => new DummyRegressor(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }
    
    /**
     * @test
     */
    public function trainPredictFeatureImportances() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);

        $importances = $this->estimator->featureImportances();

        $this->assertCount(3, $importances);
        $this->assertEquals(1.0, array_sum($importances));
    }
    
    /**
     * @test
     */
    public function trainUnlabeled() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Unlabeled::quick());
    }
    
    /**
     * @test
     */
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Unlabeled::quick([['bad']]));
    }
    
    /**
     * @test
     */
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

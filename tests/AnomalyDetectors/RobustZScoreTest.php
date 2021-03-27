<?php

namespace Rubix\ML\Tests\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\AnomalyDetectors\Scoring;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\AnomalyDetectors\RobustZScore;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

/**
 * @group AnomalyDetectors
 * @covers \Rubix\ML\AnomalyDetectors\RobustZScore
 */
class RobustZScoreTest extends TestCase
{
    /**
     * The number of samples in the training set.
     *
     * @var int
     */
    protected const TRAIN_SIZE = 300;

    /**
     * The number of samples in the validation set.
     *
     * @var int
     */
    protected const TEST_SIZE = 20;

    /**
     * The minimum validation score required to pass the test.
     *
     * @var float
     */
    protected const MIN_SCORE = 0.9;

    /**
     * Constant used to see the random number generator.
     *
     * @var int
     */
    protected const RANDOM_SEED = 0;

    /**
     * @var \Rubix\ML\Datasets\Generators\Agglomerate
     */
    protected $generator;

    /**
     * @var \Rubix\ML\AnomalyDetectors\RobustZScore
     */
    protected $estimator;

    /**
     * @var \Rubix\ML\CrossValidation\Metrics\FBeta
     */
    protected $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Agglomerate([
            0 => new Blob([0.0, 0.0], 0.5),
            1 => new Circle(0.0, 0.0, 8.0, 0.1),
        ], [0.9, 0.1]);

        $this->estimator = new RobustZScore(3.5, 0.5, 1e-9);

        $this->metric = new FBeta();

        srand(self::RANDOM_SEED);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(RobustZScore::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Scoring::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    /**
     * @test
     */
    public function badThreshold() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RobustZScore(-3.5);
    }

    /**
     * @test
     */
    public function badAlpha() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RobustZScore(3.5, 1.5);
    }

    /**
     * @test
     */
    public function type() : void
    {
        $this->assertEquals(EstimatorType::anomalyDetector(), $this->estimator->type());
    }

    /**
     * @test
     */
    public function compatibility() : void
    {
        $expected = [
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
            'threshold' => 3.5,
            'beta' => 0.5,
            'smoothing' => 1e-9,
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    /**
     * @test
     */
    public function trainPredict() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $medians = $this->estimator->medians();

        $this->assertIsArray($medians);
        $this->assertCount(2, $medians);
        $this->assertContainsOnly('float', $medians);

        $mads = $this->estimator->mads();

        $this->assertIsArray($mads);
        $this->assertCount(2, $mads);
        $this->assertContainsOnly('float', $mads);

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
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

    protected function assertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }
}

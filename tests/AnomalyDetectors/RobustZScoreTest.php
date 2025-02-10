<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\AnomalyDetectors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\AnomalyDetectors\RobustZScore;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('AnomalyDetectors')]
#[CoversClass(RobustZScore::class)]
class RobustZScoreTest extends TestCase
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

    protected RobustZScore $estimator;

    protected FBeta $metric;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                new Blob(
                    center: [0.0, 0.0],
                    stdDev: 2.0
                ),
                new Circle(
                    x: 0.0,
                    y: 0.0,
                    scale: 8.0,
                    noise: 1.0
                )
            ],
            weights: [0.9, 0.1]
        );

        $this->estimator = new RobustZScore(
            threshold: 2.0,
            beta: 0.5,
            smoothing: 1e-9
        );

        $this->metric = new FBeta();

        srand(self::RANDOM_SEED);
    }

    public function testAssertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    public function testBadThreshold() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RobustZScore(threshold: -3.5);
    }

    public function testBadBeta() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RobustZScore(threshold: 3.5, beta: 1.5);
    }

    public function testBadSmoothing() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RobustZScore(threshold: 3.5, beta: 0.5, smoothing: -1);
    }

    public function testType() : void
    {
        $this->assertEquals(EstimatorType::anomalyDetector(), $this->estimator->type());
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
            'threshold' => 2.0,
            'beta' => 0.5,
            'smoothing' => 1e-9,
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    public function testTrainPredict() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $medians = $this->estimator->medians();

        $this->assertIsArray($medians);
        $this->assertCount(2, $medians);
        $this->assertContainsOnlyFloat($medians);

        $mads = $this->estimator->mads();

        $this->assertIsArray($mads);
        $this->assertCount(2, $mads);
        $this->assertContainsOnlyFloat($mads);

        $predictions = $this->estimator->predict($testing);

        /** @var list<int|string> $labels */
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

        $this->estimator->train(Unlabeled::quick([['bad']]));
    }

    public function testPredictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

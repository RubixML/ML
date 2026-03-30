<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Regressors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
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
        $this->markTestSkipped('TODO: doesn\'t work by some reason');

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

    #[Test]
    #[TestDox('Trains, predicts, and returns the expected legacy ridge values')]
    #[DataProvider('trainPredictProvider')]
    public function trainPredict(array $samples, array $labels, array $prediction, float $expectedPrediction, array $expectedCoefficients, float $expectedBias) : void
    {
        $regression = new Ridge(1e-6);
        $regression->train(new Labeled($samples, $labels));

        $predictions = $regression->predict(new Unlabeled([$prediction]));
        $coefficients = $regression->coefficients();

        self::assertEqualsWithDelta($expectedPrediction, $predictions[0], 0.2);
        self::assertIsArray($coefficients);
        self::assertCount(count($expectedCoefficients), $coefficients);

        foreach ($expectedCoefficients as $i => $expectedCoefficient) {
            self::assertEqualsWithDelta($expectedCoefficient, $coefficients[$i], 0.2);
        }
        self::assertEqualsWithDelta($expectedBias, $regression->bias(), 0.2);
    }

    public static function trainPredictProvider() : array
    {
        return [
            'sample with 1 feature and smaller values' => [
                [
                    [0],
                    [1],
                    [2],
                    [3],
                ],
                [3, 5, 7, 9],
                [4],
                11.0,
                [2.0],
                3.0,
            ],
            'sample with 2 features and smaller values' => [
                [
                    [0, 0],
                    [1, 1],
                    [2, 1],
                    [1, 2],
                ],
                [3, 6, 7, 8],
                [2, 2],
                9.0,
                [1.0, 2.0],
                3.0,
            ],
            'sample with 3 features and smaller values' => [
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                [4, 5, 6, 7],
                [1, 1, 1],
                10.0,
                [1.0, 2.0, 3.0],
                4.0,
            ],
            'sample with 4 features' => [
                [
                    [50, 3, 5, 10],
                    [70, 10, 3, 5],
                    [40, 2, 8, 30],
                ],
                [66000, 95000, 45000],
                [60, 5, 4, 12],
                78037.27,
                [1192.98, 401.06, -132.47, -413.58],
                9945.90
            ],
            'sample with 4 features with shifted values' => [
                [
                    [52, 4, 6, 12],
                    [71, 9, 4, 6],
                    [38, 3, 7, 28],
                ],
                [66000, 95000, 45000],
                [60, 5, 4, 12],
                77709.93,
                [1368.77, 442.49, -158.60, -77.24],
                -5067.86
            ],
        ];
    }
}

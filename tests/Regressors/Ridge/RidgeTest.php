<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Regressors\Ridge;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\DataType;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Hyperplane\Hyperplane;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\EstimatorType;
use Rubix\ML\Regressors\Ridge\Ridge;

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

    public static function trainPredictProvider() : array
    {
        $isArm = in_array(strtolower(php_uname('m')), ['arm64', 'aarch64'], true);

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
                $isArm ? 77676.53 : 77644.0,
                $isArm
                    ? [1208.26, 360.18, -96.53, -420.41]
                    : [1172.0, 452.0, -70.0, -424.0],
                $isArm ? 8810.75 : 10432.0,
            ],
            'sample with 4 features with shifted values' => [
                [
                    [52, 4, 6, 12],
                    [71, 9, 4, 6],
                    [38, 3, 7, 28],
                ],
                [66000, 95000, 45000],
                [60, 5, 4, 12],
                $isArm ? 77585.35 : 78540.0,
                $isArm
                    ? [1364.07, 476.45, -161.59, -82.90]
                    : [1366.0, 504.0, -156.0, -91.0],
                $isArm ? -4999.93 : -4224.0,
            ],
        ];
    }

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

    #[Test]
    #[TestDox('Is not trained before training')]
    public function preConditions() : void
    {
        self::assertFalse($this->estimator->trained());
    }

    #[Test]
    #[TestDox('Throws when L2 penalty is invalid')]
    public function badL2Penalty() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Ridge(-1e-4);
    }

    #[Test]
    #[TestDox('Returns estimator type')]
    public function type() : void
    {
        self::assertEquals(EstimatorType::regressor(), $this->estimator->type());
    }

    #[Test]
    #[TestDox('Declares feature compatibility')]
    public function compatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        self::assertEquals($expected, $this->estimator->compatibility());
    }

    #[Test]
    #[TestDox('Trains, predicts, and returns importances')]
    public function trainPredictImportances() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        self::assertTrue($this->estimator->trained());

        $coefficients = $this->estimator->coefficients();

        self::assertIsArray($coefficients);
        self::assertCount(4, $coefficients);

        self::assertIsFloat($this->estimator->bias());

        $importances = $this->estimator->featureImportances();

        self::assertCount(4, $importances);
        self::assertContainsOnlyFloat($importances);

        $predictions = $this->estimator->predict($testing);

        /** @var list<int|float> $labels */
        $labels = $testing->labels();
        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        self::assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    #[TestDox('Throws when training set is incompatible')]
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick(samples: [['bad']], labels: [2]));
    }

    #[Test]
    #[TestDox('Throws when predicting before training')]
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }

    #[Test]
    #[TestDox('Trains, predicts, and returns the expected NumPower ridge values')]
    #[DataProvider('trainPredictProvider')]
    public function trainPredict(array $samples, array $labels, array $prediction, float $expectedPrediction, array $expectedCoefficients, float $expectedBias) : void
    {
        $regression = new Ridge(0.01);
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
}

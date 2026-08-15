<?php

declare(strict_types = 1);

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
use Rubix\ML\Regressors\Ridge;
use Rubix\ML\Tests\DataProvider\RidgeProvider;

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
        mt_srand(self::RANDOM_SEED);
    }

    // happy path

    #[Test]
    #[TestDox('[happy path] Is not trained before training')]
    public function preConditions() : void
    {
        self::assertFalse($this->estimator->trained());
    }

    #[Test]
    #[TestDox('[happy path] Returns estimator type')]
    public function type() : void
    {
        self::assertEquals(EstimatorType::regressor(), $this->estimator->type());
    }

    #[Test]
    #[TestDox('[happy path] Declares feature compatibility')]
    public function compatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        self::assertEquals($expected, $this->estimator->compatibility());
    }

    #[Test]
    #[TestDox('[happy path] Exposes params and string representation')]
    public function paramsAndToString() : void
    {
        $expected = [
            'l2 penalty' => 1.0,
        ];

        self::assertSame($expected, $this->estimator->params());
        self::assertStringContainsString('Ridge (l2 penalty: 1)', (string) $this->estimator);
    }

    #[Test]
    #[TestDox('[happy path] Trains, predicts, and returns importances')]
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

    // boundary

    #[Test]
    #[TestDox('[boundary] Coefficients and bias are null before training')]
    public function nullStateBeforeTraining() : void
    {
        self::assertNull($this->estimator->coefficients());
        self::assertNull($this->estimator->bias());
    }

    #[Test]
    #[TestDox('[boundary] Allows zero L2 penalty')]
    public function allowsZeroPenalty() : void
    {
        $regression = new Ridge(0.0);

        $regression->train(Labeled::quick(
            samples: [[0.0], [1.0], [2.0], [3.0]],
            labels: [1.0, 3.0, 5.0, 7.0]
        ));

        $predictions = $regression->predict(Unlabeled::quick([[4.0]]));

        self::assertTrue($regression->trained());
        self::assertEqualsWithDelta(8.9999980, (float) $predictions[0], 1e-7);
    }

    #[Test]
    #[TestDox('[boundary] Trains and predicts with one sample and one feature')]
    public function oneSampleOneFeature() : void
    {
        $regression = new Ridge(1.0);

        $regression->train(Labeled::quick(
            samples: [[2.0]],
            labels: [5.0]
        ));

        $predictions = $regression->predict(Unlabeled::quick([[2.0]]));

        self::assertCount(1, $predictions);
        self::assertIsFloat((float) $predictions[0]);
        self::assertTrue(is_finite((float) $predictions[0]));
    }

    // invalid input

    #[Test]
    #[TestDox('[invalid input] Throws when L2 penalty is invalid')]
    public function badL2Penalty() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Ridge(-1e-4);
    }

    #[Test]
    #[TestDox('[invalid input] Throws when training set is incompatible')]
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick(samples: [['bad']], labels: [2]));
    }

    #[Test]
    #[TestDox('[invalid input] Throws when labels are missing from training set')]
    public function trainUnlabeledDataset() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Unlabeled::quick([[1.0, 2.0, 3.0, 4.0]]));
    }

    #[Test]
    #[TestDox('[invalid input] Throws when training set is empty')]
    public function trainEmptyDataset() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick([], []));
    }

    #[Test]
    #[TestDox('[invalid input] Throws when predicting before training')]
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }

    #[Test]
    #[TestDox('[invalid input] Throws when requesting importances before training')]
    public function featureImportancesUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->featureImportances();
    }

    #[Test]
    #[TestDox('[invalid input] Throws when prediction dimensionality is incorrect')]
    public function predictDimensionalityMismatch() : void
    {
        $this->estimator->train(Labeled::quick(
            samples: [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
            labels: [4.0, 6.0, 8.0]
        ));

        $this->expectException(InvalidArgumentException::class);

        $this->estimator->predict(Unlabeled::quick([[1.0]]));
    }

    // malicious input

    #[Test]
    #[TestDox('[malicious input] Rejects script-like payload in samples')]
    public function rejectsScriptLikeSamplePayload() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::build(
            samples: [[1.0, 2.0], [3.0, '<script>alert(1)</script>']],
            labels: [1.0, 2.0]
        ));
    }

    #[Test]
    #[TestDox('[malicious input] Rejects categorical label injection')]
    public function rejectsCategoricalLabelPayload() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick(
            samples: [[1.0, 2.0], [2.0, 3.0]],
            labels: ['DROP TABLE', 'rm -rf']
        ));
    }

    // race conditions

    #[Test]
    #[TestDox('[race conditions] Predict is deterministic and does not mutate dataset')]
    public function deterministicPredictAndNoMutation() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $samplesBefore = $testing->samples();
        $labelsBefore = $testing->labels();

        $predictionsA = $this->estimator->predict($testing);
        $predictionsB = $this->estimator->predict($testing);

        self::assertCount($testing->numSamples(), $predictionsA);
        self::assertCount($testing->numSamples(), $predictionsB);
        self::assertEquals($samplesBefore, $testing->samples());
        self::assertEquals($labelsBefore, $testing->labels());

        foreach ($predictionsA as $i => $prediction) {
            self::assertEqualsWithDelta((float) $prediction, (float) $predictionsB[$i], 1e-12);
        }
    }

    #[Test]
    #[TestDox('[race conditions] Retraining overwrites previous model state')]
    public function retrainingOverwritesState() : void
    {
        $first = Labeled::quick(
            samples: [[0.0], [1.0], [2.0], [3.0]],
            labels: [1.0, 3.0, 5.0, 7.0]
        );
        $second = Labeled::quick(
            samples: [[0.0], [1.0], [2.0], [3.0]],
            labels: [7.0, 5.0, 3.0, 1.0]
        );

        $probe = Unlabeled::quick([[4.0]]);

        $this->estimator->train($first);
        $firstPrediction = $this->estimator->predict($probe)[0];

        $this->estimator->train($second);
        $secondPrediction = $this->estimator->predict($probe)[0];

        self::assertNotEqualsWithDelta((float) $firstPrediction, (float) $secondPrediction, 1.0);
    }

    // regression

    #[Test]
    #[TestDox('[regression] Trains, predicts, and returns expected NumPower ridge values')]
    #[DataProviderExternal(RidgeProvider::class, 'trainPredictProviderForNumPower')]
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

    #[Test]
    #[TestDox('[regression] Serialization preserves predictions and parameters')]
    public function serializationRegression() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(64);

        $this->estimator->train($training);

        $predictionsBefore = $this->estimator->predict($testing);
        $copy = unserialize(serialize($this->estimator));

        self::assertInstanceOf(Ridge::class, $copy);
        self::assertTrue($copy->trained());
        self::assertSame($this->estimator->params(), $copy->params());
        self::assertEquals($this->estimator->coefficients(), $copy->coefficients());
        self::assertEqualsWithDelta((float) $this->estimator->bias(), (float) $copy->bias(), 1e-12);

        $predictionsAfter = $copy->predict($testing);

        foreach ($predictionsBefore as $i => $prediction) {
            self::assertEqualsWithDelta((float) $prediction, (float) $predictionsAfter[$i], 1e-12);
        }
    }

    // property based

    #[Test]
    #[TestDox('[property based] Larger L2 penalty shrinks coefficient norm')]
    public function largerPenaltyShrinksCoefficientNorm() : void
    {
        for ($seed = 1; $seed <= 10; ++$seed) {
            [$samples, $labels] = $this->makeRandomLinearProblem(samples: 64, features: 4, seed: $seed);

            $lowPenalty = new Ridge(1e-8);
            $highPenalty = new Ridge(100.0);

            $dataset = Labeled::quick($samples, $labels);

            $lowPenalty->train($dataset);
            $highPenalty->train($dataset);

            $lowNorm = $this->l2Norm($lowPenalty->coefficients() ?? []);
            $highNorm = $this->l2Norm($highPenalty->coefficients() ?? []);

            self::assertLessThanOrEqual($lowNorm + 1e-7, $highNorm);
        }
    }

    #[Test]
    #[TestDox('[property based] Recover near-perfect linear relationships across random seeds')]
    public function recoverLinearRelationshipsAcrossSeeds() : void
    {
        for ($seed = 11; $seed <= 20; ++$seed) {
            [$samples, $labels] = $this->makeRandomLinearProblem(samples: 96, features: 3, seed: $seed);

            $regression = new Ridge(1e-8);
            $regression->train(Labeled::quick($samples, $labels));

            $predictions = $regression->predict(Unlabeled::quick($samples));

            $score = $this->metric->score(predictions: $predictions, labels: $labels);

            self::assertGreaterThanOrEqual(0.999999, $score);
        }
    }

    // fuzzing

    #[Test]
    #[TestDox('[fuzzing] Random numeric datasets produce finite predictions')]
    public function randomDatasetsProduceFinitePredictions() : void
    {
        mt_srand(1337);

        for ($iteration = 0; $iteration < 25; ++$iteration) {
            $features = mt_rand(1, 8);
            $samplesCount = mt_rand(8, 48);
            $testCount = mt_rand(1, 16);

            [$samples, $labels] = $this->makeRandomLinearProblem(
                samples: $samplesCount,
                features: $features,
                seed: 2000 + $iteration
            );
            [$testSamples] = $this->makeRandomLinearProblem(
                samples: $testCount,
                features: $features,
                seed: 3000 + $iteration
            );

            $penalty = mt_rand(0, 1000) / 10.0;
            $regression = new Ridge($penalty);

            $regression->train(Labeled::quick($samples, $labels));
            $predictions = $regression->predict(Unlabeled::quick($testSamples));

            self::assertCount($testCount, $predictions);

            foreach ($predictions as $prediction) {
                self::assertIsNumeric($prediction);
                self::assertTrue(is_finite((float) $prediction));
            }
        }
    }

    /**
     * Make random linear problem
     *
     * @param int $samples
     * @param int $features
     * @param int $seed
     * @return array{0: list<list<float>>, 1: list<float>}
     */
    private function makeRandomLinearProblem(int $samples, int $features, int $seed) : array
    {
        mt_srand($seed);

        $coefficients = [];

        for ($j = 0; $j < $features; ++$j) {
            $coefficients[] = mt_rand(-200, 200) / 10.0;
        }

        $bias = mt_rand(-100, 100) / 10.0;

        $x = [];
        $y = [];

        for ($i = 0; $i < $samples; ++$i) {
            $sample = [];

            for ($j = 0; $j < $features; ++$j) {
                $sample[] = mt_rand(-500, 500) / 10.0;
            }

            $target = $bias;

            foreach ($sample as $j => $value) {
                $target += $value * $coefficients[$j];
            }

            $x[] = $sample;
            $y[] = $target;
        }

        return [$x, $y];
    }

    /**
     * @param (int|float)[] $values
     */
    private function l2Norm(array $values) : float
    {
        $sum = 0.0;

        foreach ($values as $value) {
            $sum += (float) $value * (float) $value;
        }

        return sqrt($sum);
    }
}

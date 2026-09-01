<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Clusterers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Clusterers\Seeders\KMC2;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Clusterers\GaussianMixture;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

use function array_fill;
use function is_nan;
use function max;
use function min;

#[Group('Clusterers')]
#[CoversClass(GaussianMixture::class)]
class GaussianMixtureTest extends TestCase
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

    protected GaussianMixture $estimator;

    protected VMeasure $metric;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'red' => new Blob(
                    center: [255, 32, 0],
                    stdDev: 50.0
                ),
                'green' => new Blob(
                    center: [0, 128, 0],
                    stdDev: 10.0
                ),
                'blue' => new Blob(
                    center: [0, 32, 255],
                    stdDev: 30.0
                ),
            ],
            weights: [0.5, 0.2, 0.3]
        );

        $this->estimator = new GaussianMixture(
            k: 3,
            smoothing: 1e-9,
            epochs: 100,
            minChange: 1e-3,
            seeder: new KMC2(m: 50)
        );

        $this->metric = new VMeasure();

        srand(self::RANDOM_SEED);
    }

    #[Test]
    public function preConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    #[Test]
    public function badK() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new GaussianMixture(k: 0);
    }

    #[Test]
    public function type() : void
    {
        $this->assertEquals(EstimatorType::clusterer(), $this->estimator->type());
    }

    #[Test]
    public function compatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    #[Test]
    public function params() : void
    {
        $expected = [
            'k' => 3,
            'smoothing' => 1e-9,
            'epochs' => 100,
            'min change' => 1e-3,
            'seeder' => new KMC2(50),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    public function trainPredict() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $priors = $this->estimator->priors();

        $this->assertIsArray($priors);
        $this->assertCount(3, $priors);
        $this->assertContainsOnlyFloat($priors);

        $means = $this->estimator->means();

        $this->assertIsArray($means);
        $this->assertCount(3, $means);
        $this->assertContainsOnlyArray($means);

        $variances = $this->estimator->variances();

        $this->assertIsArray($variances);
        $this->assertCount(3, $variances);
        $this->assertContainsOnlyArray($variances);

        $losses = $this->estimator->losses();

        $this->assertIsArray($losses);
        $this->assertContainsOnlyFloat($losses);

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $testing->labels()
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    public function trainHighDimensional() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $d = 60;

        $generator = new Agglomerate([
            'a' => new Blob(array_fill(0, $d, 0.0), 2000.0),
            'b' => new Blob(array_fill(0, $d, 1e6), 2000.0),
            'c' => new Blob(array_fill(0, $d, -1e6), 2000.0),
        ]);

        $training = $generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        foreach ($this->estimator->means() as $means) {
            $this->assertNotContainsNAN($means);
        }

        foreach ($this->estimator->variances() as $variances) {
            $this->assertNotContainsNAN($variances);
        }

        $losses = $this->estimator->losses();

        $this->assertIsArray($losses);

        $this->assertNotContainsNAN($losses);
    }

    #[Test]
    public function trainDiverseClusterScales() : void
    {
        $generator = new Agglomerate([
            'wide' => new Blob(center: [0.0, 0.0], stdDev: 10000.0),
            'tight' => new Blob(center: [5000.0, 5000.0], stdDev: 0.1),
        ]);

        $estimator = new GaussianMixture(
            k: 2,
            smoothing: 1e-9,
            epochs: 100,
            minChange: 1e-3,
            seeder: new KMC2(m: 50)
        );

        $estimator->setLogger(new BlackHole());

        $estimator->train($generator->generate(self::TRAIN_SIZE));

        $tightVariance = INF;

        foreach ($estimator->variances() as $variances) {
            $tightVariance = min($tightVariance, max($variances));
        }

        $this->assertLessThan(0.05, $tightVariance);
    }

    #[Test]
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Unlabeled::quick(samples: [['bad']]));
    }

    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }

    /**
     * @param (int|float)[] $values
     */
    protected function assertNotContainsNAN(array $values) : void
    {
        foreach ($values as $value) {
            $this->assertFalse(is_nan($value), 'Value must not be NAN.');
        }
    }
}

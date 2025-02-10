<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Clusterers;

use PHPUnit\Framework\Attributes\CoversClass;
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

    public function testAssertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    public function testBadK() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new GaussianMixture(k: 0);
    }

    public function testType() : void
    {
        $this->assertEquals(EstimatorType::clusterer(), $this->estimator->type());
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
            'k' => 3,
            'smoothing' => 1e-9,
            'epochs' => 100,
            'min change' => 1e-3,
            'seeder' => new KMC2(50),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    public function testTrainPredict() : void
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
}

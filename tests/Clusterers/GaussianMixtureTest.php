<?php

namespace Rubix\ML\Tests\Clusterers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
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

/**
 * @group Clusterers
 * @covers \Rubix\ML\Clusterers\GaussianMixture
 */
class GaussianMixtureTest extends TestCase
{
    /**
     * The number of samples in the training set.
     *
     * @var int
     */
    protected const TRAIN_SIZE = 512;

    /**
     * The number of samples in the validation set.
     *
     * @var int
     */
    protected const TEST_SIZE = 256;

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
     * @var Agglomerate
     */
    protected $generator;

    /**
     * @var GaussianMixture
     */
    protected $estimator;

    /**
     * @var VMeasure
     */
    protected $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Agglomerate([
            'red' => new Blob([255, 32, 0], 50.0),
            'green' => new Blob([0, 128, 0], 10.0),
            'blue' => new Blob([0, 32, 255], 30.0),
        ], [0.5, 0.2, 0.3]);

        $this->estimator = new GaussianMixture(3, 1e-9, 100, 1e-3, new KMC2(50));

        $this->metric = new VMeasure();

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
        $this->assertInstanceOf(GaussianMixture::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Verbose::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    /**
     * @test
     */
    public function badK() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new GaussianMixture(0);
    }

    /**
     * @test
     */
    public function type() : void
    {
        $this->assertEquals(EstimatorType::clusterer(), $this->estimator->type());
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
            'k' => 3,
            'smoothing' => 1e-9,
            'epochs' => 100,
            'min change' => 1e-3,
            'seeder' => new KMC2(50),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    /**
     * @test
     */
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
        $this->assertContainsOnly('float', $priors);

        $means = $this->estimator->means();

        $this->assertIsArray($means);
        $this->assertCount(3, $means);
        $this->assertContainsOnly('array', $means);

        $variances = $this->estimator->variances();

        $this->assertIsArray($variances);
        $this->assertCount(3, $variances);
        $this->assertContainsOnly('array', $variances);

        $losses = $this->estimator->losses();

        $this->assertIsArray($losses);
        $this->assertContainsOnly('float', $losses);

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    /**
     * @test
     */
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

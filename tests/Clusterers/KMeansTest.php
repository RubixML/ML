<?php

namespace Rubix\ML\Tests\Clusterers;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Other\Loggers\BlackHole;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Clusterers\Seeders\PlusPlus;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

/**
 * @group Clusterers
 * @covers \Rubix\ML\Clusterers\KMeans
 */
class KMeansTest extends TestCase
{
    protected const TRAIN_SIZE = 450;
    protected const TEST_SIZE = 10;
    protected const MIN_SCORE = 0.9;

    protected const RANDOM_SEED = 0;

    /**
     * @var \Rubix\ML\Datasets\Generators\Agglomerate
     */
    protected $generator;

    /**
     * @var \Rubix\ML\Clusterers\KMeans
     */
    protected $estimator;

    /**
     * @var \Rubix\ML\CrossValidation\Metrics\VMeasure
     */
    protected $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Agglomerate([
            'red' => new Blob([255, 32, 0], 30.0),
            'green' => new Blob([0, 128, 0], 10.0),
            'blue' => new Blob([0, 32, 255], 20.0),
        ], [2, 3, 4]);

        $this->estimator = new KMeans(3, 100, 300, 1e-4, 5, new Euclidean(), new PlusPlus());

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
        $this->assertInstanceOf(KMeans::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Online::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Verbose::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    /**
     * @test
     */
    public function badK() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new KMeans(0);
    }

    /**
     * @test
     */
    public function type() : void
    {
        $this->assertSame(Estimator::CLUSTERER, $this->estimator->type());
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
            'batch_size' => 100,
            'epochs' => 300,
            'min_change' => 1e-4,
            'window' => 5,
            'kernel' => new Euclidean(),
            'seeder' => new PlusPlus(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    /**
     * @test
     */
    public function trainPartialPredict() : void
    {
        $this->estimator->setLogger(new BlackHole());
        
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $folds = $training->stratifiedFold(3);

        $this->estimator->train($folds[0]);
        $this->estimator->partial($folds[1]);
        $this->estimator->partial($folds[2]);

        $this->assertTrue($this->estimator->trained());

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

        $this->estimator->predict(Unlabeled::quick([[1.0]]));
    }
}

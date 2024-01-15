<?php

namespace Rubix\ML\Tests\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

/**
 * @group Classifiers
 * @covers \Rubix\ML\Classifiers\KNearestNeighbors
 */
class KNearestNeighborsTest extends TestCase
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
     * @var KNearestNeighbors
     */
    protected $estimator;

    /**
     * @var FBeta
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

        $this->estimator = new KNearestNeighbors(3, true, new Euclidean());

        $this->metric = new FBeta();

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
        $this->assertInstanceOf(KNearestNeighbors::class, $this->estimator);
        $this->assertInstanceOf(Online::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    /**
     * @test
     */
    public function badK() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new KNearestNeighbors(0);
    }

    /**
     * @test
     */
    public function type() : void
    {
        $this->assertEquals(EstimatorType::classifier(), $this->estimator->type());
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
            'weighted' => true,
            'kernel' => new Euclidean(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    /**
     * @test
     */
    public function trainPartialPredict() : void
    {
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

        $this->estimator->train(Labeled::quick([['bad']], ['green']));
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

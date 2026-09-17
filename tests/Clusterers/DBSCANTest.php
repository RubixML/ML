<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Clusterers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Clusterers\DBSCAN;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\VMeasure;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Clusterers')]
#[CoversClass(DBSCAN::class)]
class DBSCANTest extends TestCase
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

    protected DBSCAN $estimator;

    protected VMeasure $metric;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            [
                'inner' => new Circle(x: 0.0, y: 0.0, scale: 1.0, noise: 0.01),
                'middle' => new Circle(x: 0.0, y: 0.0, scale: 5.0, noise: 0.05),
                'outer' => new Circle(x: 0.0, y: 0.0, scale: 10.0, noise: 0.1),
            ]
        );

        $this->estimator = new DBSCAN(radius: 3.0, minDensity: 10, tree: new BallTree());

        $this->metric = new VMeasure();

        srand(self::RANDOM_SEED);
    }

    #[Test]
    public function preConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    #[Test]
    public function badRadius() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new DBSCAN(radius: 0.0);
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
            'radius' => 3.0,
            'min density' => 10,
            'tree' => new BallTree(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    public function trainPredict() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $testing->labels()
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    public function predictsNoiseForIsolatedSamples() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $this->assertSame(
            [DBSCAN::NOISE],
            $this->estimator->predict(Unlabeled::quick(samples: [[0.0, 100.0]]))
        );
    }

    #[Test]
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Unlabeled::quick(samples: [['bad']]));
    }

    #[Test]
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick(samples: [[1.0, 2.0]]));
    }

    #[Test]
    #[TestDox('Throws an exception when predicting with incompatible data')]
    public function predictIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $this->estimator->predict(Unlabeled::quick(samples: [[1.0]]));
    }

    #[Test]
    public function restoreStateFromSerializedModel() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $restored = unserialize(serialize($this->estimator));

        $this->assertTrue($restored->trained());

        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->assertEquals($this->estimator->predict($testing), $restored->predict($testing));
    }
}

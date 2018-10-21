<?php

namespace Rubix\ML\Tests\AnomalyDetectors;

use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\AnomalyDetectors\RobustZScore;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class RobustZScoreTest extends TestCase
{
    const TRAIN_SIZE = 300;
    const TEST_SIZE = 5;

    protected $generator;

    protected $estimator;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            0 => new Blob([0., 0.], 0.1),
            1 => new Circle(0., 0., 8., 0.1),
        ], [0.8, 0.2]);

        $this->estimator = new RobustZScore(1.25, 1.5);
    }

    public function test_build_detector()
    {
        $this->assertInstanceOf(RobustZScore::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::DETECTOR, $this->estimator->type());
    }

    public function test_train_predict()
    {
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($this->generator->generate(self::TRAIN_SIZE));

        foreach ($this->estimator->predict($testing) as $i => $prediction) {
            $this->assertEquals($testing->label($i), $prediction);
        }
    }

    public function test_predict_untrained()
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

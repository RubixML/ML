<?php

namespace Rubix\ML\Tests\AnomalyDetectors;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\AnomalyDetectors\LocalOutlierFactor;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class LocalOutlierFactorTest extends TestCase
{
    const TRAIN_SIZE = 300;
    const TEST_SIZE = 10;

    protected $generator;

    protected $estimator;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            0 => new Blob([0., 0.], 0.5),
            1 => new Circle(0., 0., 8., 0.1),
        ], [0.9, 0.1]);

        $this->estimator = new LocalOutlierFactor(20, 0.1, new Euclidean());
    }

    public function test_build_detector()
    {
        $this->assertInstanceOf(LocalOutlierFactor::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Online::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::DETECTOR, $this->estimator->type());
    }

    public function test_train_partial_predict()
    {
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($this->generator->generate(self::TRAIN_SIZE / 3));
        $this->estimator->partial($this->generator->generate(self::TRAIN_SIZE / 3));
        $this->estimator->partial($this->generator->generate(self::TRAIN_SIZE / 3));

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

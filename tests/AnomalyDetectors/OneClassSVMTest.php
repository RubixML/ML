<?php

namespace Rubix\ML\Tests\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Kernels\SVM\Polynomial;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\AnomalyDetectors\OneClassSVM;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class OneClassSVMTest extends TestCase
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

        $this->estimator = new OneClassSVM(0.05, new Polynomial(4, 1e-3), true, 1e-4);
    }

    public function test_build_detector()
    {
        $this->assertInstanceOf(OneClassSVM::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
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

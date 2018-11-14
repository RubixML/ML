<?php

namespace Rubix\ML\Tests\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Regressors\SVR;
use Rubix\ML\Kernels\SVM\RBF;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\SwissRoll;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class SVRTest extends TestCase
{
    const TRAIN_SIZE = 300;
    const TEST_SIZE = 5;
    const TOLERANCE = 5;

    protected $generator;

    protected $estimator;

    public function setUp()
    {
        $this->generator = new SwissRoll(4., -7., 0., 1., 0.2);

        $this->estimator = new SVR(1.0, 0.025, new RBF(1e-3), true, 1e-3);
    }

    public function test_build_regressor()
    {
        $this->assertInstanceOf(SVR::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::REGRESSOR, $this->estimator->type());
    }

    public function test_train_predict()
    {
        $this->estimator->train($this->generator->generate(self::TRAIN_SIZE));

        $testing = $this->generator->generate(self::TEST_SIZE);

        foreach ($this->estimator->predict($testing) as $i => $prediction) {
            $this->assertEquals($testing->label($i), $prediction, '', self::TOLERANCE);
        }
    }

    public function test_train_with_unlabeled()
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Unlabeled::quick());
    }

    public function test_predict_untrained()
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

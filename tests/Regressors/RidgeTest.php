<?php

namespace Rubix\ML\Tests\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Regressors\Ridge;
use Rubix\ML\Datasets\Unlabeled;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class RidgeTest extends TestCase
{
    const TOLERANCE = 3;
    
    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $this->training = unserialize(file_get_contents(dirname(__DIR__) . '/mpg.dataset') ?: '');

        $this->testing = $this->training->randomize()->head(3);

        $this->estimator = new Ridge(1.0);
    }

    public function test_build_regressor()
    {
        $this->assertInstanceOf(Ridge::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::REGRESSOR, $this->estimator->type());
    }

    public function test_make_prediction()
    {
        $this->estimator->train($this->training);

        foreach ($this->estimator->predict($this->testing) as $i => $prediction) {
            $this->assertEquals($this->testing->label($i), $prediction, '', self::TOLERANCE);
        }
    }

    public function test_train_with_unlabeled()
    {
        $dataset = new Unlabeled([['bad']]);

        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train($dataset);
    }

    public function test_predict_untrained()
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict($this->testing);
    }
}

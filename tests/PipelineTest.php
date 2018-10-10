<?php

namespace Rubix\ML\Tests;

use Rubix\ML\Online;
use Rubix\ML\Pipeline;
use Rubix\ML\Persistable;
use Rubix\ML\MetaEstimator;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\NumericStringConverter;
use PHPUnit\Framework\TestCase;

class PipelineTest extends TestCase
{
    protected $estimator;

    protected $training;

    protected $testing;

    public function setUp()
    {
        $this->training = Labeled::load(__DIR__ . '/iris.dataset');

        $this->testing = $this->training->randomize()->head(3);

        $this->estimator = new Pipeline(new GaussianNB(), [
            new NumericStringConverter(),
            new OneHotEncoder(),
            new ZScaleStandardizer(true),
        ]);
    }

    public function test_build_meta_estimator()
    {
        $this->assertInstanceOf(Pipeline::class, $this->estimator);
        $this->assertInstanceOf(MetaEstimator::class, $this->estimator);
        $this->assertInstanceOf(Online::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    public function test_make_prediction()
    {
        $this->estimator->train($this->training);

        $predictions = $this->estimator->predict($this->testing);

        $this->assertEquals($this->testing->label(0), $predictions[0]);
        $this->assertEquals($this->testing->label(1), $predictions[1]);
        $this->assertEquals($this->testing->label(2), $predictions[2]);
    }

    public function test_partial_train()
    {
        $folds = $this->training->stratifiedFold(3);

        $this->estimator->train($folds[0]);

        $this->estimator->partial($folds[1]);

        $this->estimator->partial($folds[2]);

        $predictions = $this->estimator->predict($this->testing);

        $this->assertEquals($this->testing->label(0), $predictions[0]);
        $this->assertEquals($this->testing->label(1), $predictions[1]);
        $this->assertEquals($this->testing->label(2), $predictions[2]);
    }
}

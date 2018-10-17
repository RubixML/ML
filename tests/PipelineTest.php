<?php

namespace Rubix\ML\Tests;

use Rubix\ML\Online;
use Rubix\ML\Pipeline;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\MetaEstimator;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\NaiveBayes;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Transformers\IntervalDiscretizer;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class PipelineTest extends TestCase
{
    const TRAIN_SIZE = 200;
    const TEST_SIZE = 5;
    const MIN_PROB = 0.33;

    protected $generator;

    protected $estimator;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            'red' => new Blob([255, 0, 0], 3.),
            'green' => new Blob([0, 128, 0], 1.),
            'blue' => new Blob([0, 0, 255], 2.),
        ]);

        $this->estimator = new Pipeline(new NaiveBayes(1.), [
            new IntervalDiscretizer(6),
        ]);
    }

    public function test_build_meta_estimator()
    {
        $this->assertInstanceOf(Pipeline::class, $this->estimator);
        $this->assertInstanceOf(MetaEstimator::class, $this->estimator);
        $this->assertInstanceOf(Online::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::CLASSIFIER, $this->estimator->type());
    }

    public function test_train_partial_predict_proba()
    {
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($this->generator->generate(self::TRAIN_SIZE / 2));
        $this->estimator->partial($this->generator->generate(self::TRAIN_SIZE / 2));

        foreach ($this->estimator->predict($testing) as $i => $prediction) {
            $this->assertEquals($testing->label($i), $prediction);
        }

        foreach ($this->estimator->proba($testing) as $i => $prob) {
            $this->assertGreaterThan(self::MIN_PROB, $prob[$testing->label($i)]);
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

<?php

namespace Rubix\ML\Tests\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class GaussianNBTest extends TestCase
{
    const TRAIN_SIZE = 100;
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

        $this->estimator = new GaussianNB(null);
    }

    public function test_build_classifier()
    {
        $this->assertInstanceOf(GaussianNB::class, $this->estimator);
        $this->assertInstanceOf(Online::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
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

        $this->estimator->train($this->generator->generate(self::TRAIN_SIZE / 3));
        $this->estimator->partial($this->generator->generate(self::TRAIN_SIZE / 3));
        $this->estimator->partial($this->generator->generate(self::TRAIN_SIZE / 3));

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

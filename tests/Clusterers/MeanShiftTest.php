<?php

namespace Rubix\ML\Tests\Clusterers;

use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Clusterers\MeanShift;
use Rubix\ML\Other\Loggers\BlackHole;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class MeanShiftTest extends TestCase
{
    const TEST_SIZE = 300;
    const TOLERANCE = 3;

    protected $generator;

    protected $estimator;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            'red' => new Blob([255, 0, 0], 3.),
            'green' => new Blob([0, 128, 0], 1.),
            'blue' => new Blob([0, 0, 255], 2.),
        ]);

        $this->estimator = new MeanShift(10.0, new Euclidean(), 100, 1e-4);

        $this->estimator->setLogger(new BlackHole());
    }

    public function test_build_clusterer()
    {
        $this->assertInstanceOf(MeanShift::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Verbose::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::CLUSTERER, $this->estimator->type());
    }

    public function test_train_predict()
    {
        $dataset = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($dataset);

        $predictions = $this->estimator->predict($dataset);

        $clusters = array_count_values($predictions);

        $this->assertEquals(self::TEST_SIZE / 3, $clusters[0], '', self::TOLERANCE);
        $this->assertEquals(self::TEST_SIZE / 3, $clusters[1], '', self::TOLERANCE);
        $this->assertEquals(self::TEST_SIZE / 3, $clusters[2], '', self::TOLERANCE);
    }

    public function test_predict_untrained()
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

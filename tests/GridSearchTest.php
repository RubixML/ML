<?php

namespace Rubix\ML\Tests;

use Rubix\ML\Estimator;
use Rubix\ML\GridSearch;
use Rubix\ML\Persistable;
use Rubix\ML\MetaEstimator;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\F1Score;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class GridSearchTest extends TestCase
{
    const TRAIN_SIZE = 250;
    const TEST_SIZE = 5;
    const MIN_PROB = 0.33;

    protected $generator;

    protected $estimator;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            'inner' => new Circle(0., 0., 1., 0.01),
            'middle' => new Circle(0., 0., 5., 0.05),
            'outer' => new Circle(0., 0., 10., 0.1),
        ]);

        $this->estimator = new GridSearch(KNearestNeighbors::class, [
            [1, 3, 5], [new Euclidean(), new Manhattan()],
        ], new F1Score(), new HoldOut(0.2));
    }

    public function test_build_meta_estimator()
    {
        $this->assertInstanceOf(GridSearch::class, $this->estimator);
        $this->assertInstanceOf(MetaEstimator::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::CLASSIFIER, $this->estimator->type());
    }

    public function test_train_predict_proba()
    {
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($this->generator->generate(self::TRAIN_SIZE));

        foreach ($this->estimator->predict($testing) as $i => $prediction) {
            $this->assertEquals($testing->label($i), $prediction);
        }

        foreach ($this->estimator->proba($testing) as $i => $prob) {
            $this->assertGreaterThan(self::MIN_PROB, $prob[$testing->label($i)]);
        }
    }
}

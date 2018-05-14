<?php

use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Estimators\Estimator;
use Rubix\Engine\Estimators\Regressor;
use Rubix\Engine\Persisters\Persistable;
use Rubix\Engine\Estimators\RegressionTree;
use PHPUnit\Framework\TestCase;

class RegressionTreeTest extends TestCase
{
    protected $estimator;

    protected $dataset;

    public function setUp()
    {
        $this->dataset = Supervised::fromIterator([
            [4, 91.0, 1795, 33.0],
            [6, 225.0, 3651, 20.0],
            [6, 250.0, 3574, 18.0],
            [6, 250.0, 3645, 18.5],
            [6, 258.0, 3193, 17.5],
            [4, 97.0, 1825, 29.5],
            [4, 85.0, 1990, 32.0],
            [4, 87.0, 2155, 28.0],
            [4, 130.0, 3150, 20.0],
            [8, 318.0, 3940, 13.0],
            [4, 120.0, 3270, 19.0],
            [8, 260.0, 3365, 19.9],
        ]);

        $this->estimator = new RegressionTree(3, 10);
    }

    public function test_create_tree()
    {
        $this->assertInstanceOf(RegressionTree::class, $this->estimator);
        $this->assertInstanceOf(Regressor::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    public function test_make_prediction()
    {
        $this->estimator->train($this->dataset);

        $prediction = $this->estimator->predict([6, 150.0, 2825]);

        $this->assertThat($prediction->outcome(), $this->logicalAnd($this->greaterThan(19), $this->lessThan(21)));
    }
}

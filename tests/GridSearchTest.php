<?php

use Rubix\ML\GridSearch;
use Rubix\ML\Persistable;
use Rubix\ML\MetaEstimator;
use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use PHPUnit\Framework\TestCase;

class GridSearchTest extends TestCase
{
    protected $estimator;

    public function setUp()
    {
        $this->estimator = new GridSearch(DummyClassifier::class, [], new Accuracy(), new HoldOut(0.2));
    }

    public function test_build_grid_search()
    {
        $this->assertInstanceOf(GridSearch::class, $this->estimator);
        $this->assertInstanceOf(MetaEstimator::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }
}

<?php

namespace Rubix\Tests;

use Rubix\ML\Pipeline;
use Rubix\ML\Persistable;
use Rubix\ML\MetaEstimator;
use Rubix\ML\Classifiers\DummyClassifier;
use PHPUnit\Framework\TestCase;

class PipelineTest extends TestCase
{
    protected $estimator;

    public function setUp()
    {
        $this->estimator = new Pipeline(new DummyClassifier(), []);
    }

    public function test_create_tree()
    {
        $this->assertInstanceOf(Pipeline::class, $this->estimator);
        $this->assertInstanceOf(MetaEstimator::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }
}

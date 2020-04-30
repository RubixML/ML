<?php

namespace Rubix\ML\Tests\Backends\Tasks;

use Rubix\ML\Backends\Tasks\Predict;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

/**
 * @group Tasks
 * @covers \Rubix\ML\Backends\Tasks\Predict
 */
class PredictTest extends TestCase
{
    /**
     * @test
     */
    public function compute() : void
    {
        $estimator = new DummyClassifier();

        $generator = new Agglomerate([
            'male' => new Blob([69.2, 195.7, 40.0], [1.0, 3.0, 0.3]),
            'female' => new Blob([63.7, 168.5, 38.1], [0.8, 2.5, 0.4]),
        ], [0.45, 0.55]);

        $training = $generator->generate(50);

        $estimator->train($training);

        $testing = $generator->generate(15);

        $task = new Predict($estimator, $testing);

        $result = $task->compute();

        $this->assertCount(15, $result);
    }
}

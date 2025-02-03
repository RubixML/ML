<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Backends\Tasks;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Backends\Tasks\TrainLearner;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

#[Group('Tasks')]
#[CoversClass(TrainLearner::class)]
class TrainLearnerTest extends TestCase
{
    public function testCompute() : void
    {
        $estimator = new GaussianNB();

        $generator = new Agglomerate(
            generators: [
                'male' => new Blob(
                    center: [69.2, 195.7, 40.0],
                    stdDev: [1.0, 3.0, 0.3]
                ),
                'female' => new Blob(
                    center: [63.7, 168.5, 38.1],
                    stdDev: [0.8, 2.5, 0.4]
                ),
            ],
            weights: [0.45, 0.55]
        );

        $dataset = $generator->generate(50);

        $task = new TrainLearner(estimator: $estimator, dataset: $dataset);

        $result = $task->compute();

        $this->assertInstanceOf(GaussianNB::class, $result);
        $this->assertTrue($result->trained());
    }
}

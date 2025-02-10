<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Backends\Tasks;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Backends\Tasks\TrainAndValidate;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use PHPUnit\Framework\TestCase;

#[Group('Tasks')]
#[CoversClass(TrainAndValidate::class)]
class TrainAndValidateTest extends TestCase
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

        $metric = new Accuracy();

        $training = $generator->generate(50);
        $testing = $generator->generate(15);

        $task = new TrainAndValidate(
            estimator: $estimator,
            training: $training,
            testing: $testing,
            metric: $metric
        );

        $result = $task->compute();

        $this->assertIsFloat($result);
    }
}

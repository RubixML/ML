<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Backends\Tasks;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Backends\Tasks\CrossValidate;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Datasets\Generators\Agglomerate;
use PHPUnit\Framework\TestCase;

#[Group('Tasks')]
#[CoversClass(CrossValidate::class)]
class CrossValidateTest extends TestCase
{
    #[Test]
    public function compute() : void
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

        $validator = new KFold(3);

        $metric = new Accuracy();

        $dataset = $generator->generate(60);

        $task = new CrossValidate(
            estimator: $estimator,
            dataset: $dataset,
            validator: $validator,
            metric: $metric
        );

        $result = $task->compute();

        $this->assertIsFloat($result);
        $this->assertGreaterThanOrEqual(0.0, $result);
        $this->assertLessThanOrEqual(1.0, $result);
    }
}

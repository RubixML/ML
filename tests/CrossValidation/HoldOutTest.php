<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\CrossValidation;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use PHPUnit\Framework\TestCase;

#[Group('Validators')]
#[CoversClass(HoldOut::class)]
class HoldOutTest extends TestCase
{
    protected const int DATASET_SIZE = 50;

    protected Agglomerate $generator;

    protected GaussianNB $estimator;

    protected HoldOut $validator;

    protected Accuracy $metric;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'male' => new Blob(
                    center: [69.2, 195.7, 40.],
                    stdDev: [1., 3., 0.3]
                ),
                'female' => new Blob(
                    center: [63.7, 168.5, 38.1],
                    stdDev: [0.8, 2.5, 0.4]
                ),
            ],
            weights: [0.45, 0.55]
        );

        $this->estimator = new GaussianNB();

        $this->validator = new HoldOut(0.2);

        $this->metric = new Accuracy();
    }

    public function testTestEstimator() : void
    {
        [$min, $max] = $this->metric->range()->list();

        $dataset = $this->generator->generate(self::DATASET_SIZE);

        $score = $this->validator->test(
            estimator: $this->estimator,
            dataset: $dataset,
            metric: $this->metric
        );

        $this->assertThat(
            $score,
            $this->logicalAnd(
                $this->greaterThanOrEqual($min),
                $this->lessThanOrEqual($max)
            )
        );
    }
}

<?php

namespace Rubix\ML\Tests\CrossValidation;

use Rubix\ML\CrossValidation\HoldOut;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\CrossValidation\Validator;
use Rubix\ML\Classifiers\GaussianNB;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use PHPUnit\Framework\TestCase;

/**
 * @group Validators
 * @covers \Rubix\ML\CrossValidation\HoldOut
 */
class HoldOutTest extends TestCase
{
    protected const DATASET_SIZE = 50;

    /**
     * @var Agglomerate
     */
    protected $generator;

    /**
     * @var GaussianNB
     */
    protected $estimator;

    /**
     * @var HoldOut
     */
    protected $validator;

    /**
     * @var Accuracy
     */
    protected $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Agglomerate([
            'male' => new Blob([69.2, 195.7, 40.], [1., 3., 0.3]),
            'female' => new Blob([63.7, 168.5, 38.1], [0.8, 2.5, 0.4]),
        ], [0.45, 0.55]);

        $this->estimator = new GaussianNB();

        $this->validator = new HoldOut(0.2);

        $this->metric = new Accuracy();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(HoldOut::class, $this->validator);
        $this->assertInstanceOf(Validator::class, $this->validator);
    }

    /**
     * @test
     */
    public function test() : void
    {
        [$min, $max] = $this->metric->range()->list();

        $dataset = $this->generator->generate(self::DATASET_SIZE);

        $score = $this->validator->test($this->estimator, $dataset, $this->metric);

        $this->assertThat(
            $score,
            $this->logicalAnd(
                $this->greaterThanOrEqual($min),
                $this->lessThanOrEqual($max)
            )
        );
    }
}

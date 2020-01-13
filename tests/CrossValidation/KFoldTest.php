<?php

namespace Rubix\ML\Tests\CrossValidation;

use Rubix\ML\Parallel;
use Rubix\ML\Backends\Serial;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\CrossValidation\Validator;
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use PHPUnit\Framework\TestCase;

/**
 * @group Validators
 * @covers \Rubix\ML\CrossValidation\KFold
 */
class KFoldTest extends TestCase
{
    protected const DATASET_SIZE = 50;

    /**
     * @var \Rubix\ML\Datasets\Generators\Agglomerate
     */
    protected $generator;

    /**
     * @var \Rubix\ML\Classifiers\DummyClassifier
     */
    protected $estimator;

    /**
     * @var \Rubix\ML\CrossValidation\KFold
     */
    protected $validator;

    /**
     * @var \Rubix\ML\CrossValidation\Metrics\Accuracy
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

        $this->estimator = new DummyClassifier();

        $this->validator = new KFold(10);

        $this->validator->setBackend(new Serial());

        $this->metric = new Accuracy();
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(KFold::class, $this->validator);
        $this->assertInstanceOf(Validator::class, $this->validator);
        $this->assertInstanceOf(Parallel::class, $this->validator);
    }

    /**
     * @test
     */
    public function test() : void
    {
        [$min, $max] = $this->metric->range();

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

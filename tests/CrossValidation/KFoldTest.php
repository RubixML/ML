<?php

namespace Rubix\ML\Tests\CrossValidation;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\CrossValidation\Validator;
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use PHPUnit\Framework\TestCase;

class KFoldTest extends TestCase
{
    const TRAIN_SIZE = 50;

    protected $generator;

    protected $estimator;

    protected $validator;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            'male' => new Blob([69.2, 195.7, 40.], [1., 3., 0.3]),
            'female' => new Blob([63.7, 168.5, 38.1], [0.8, 2.5, 0.4]),
        ], [0.45, 0.55]);

        $this->estimator = new DummyClassifier();

        $this->validator = new KFold(10, false);
    }

    public function test_build_validator()
    {
        $this->assertInstanceOf(KFold::class, $this->validator);
        $this->assertInstanceOf(Validator::class, $this->validator);
    }

    public function test_test_estimator()
    {
        $dataset = $this->generator->generate(self::TRAIN_SIZE);

        $score = $this->validator->test($this->estimator, $dataset, new Accuracy());

        $this->assertEquals(.5, $score, '', .5);
    }
}

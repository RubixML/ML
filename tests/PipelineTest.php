<?php

namespace Rubix\ML\Tests;

use Rubix\ML\Online;
use Rubix\ML\Wrapper;
use Rubix\ML\Verbose;
use Rubix\ML\Pipeline;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Classifiers\NaiveBayes;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\F1Score;
use Rubix\ML\Transformers\IntervalDiscretizer;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class PipelineTest extends TestCase
{
    const TRAIN_SIZE = 300;
    const TEST_SIZE = 10;
    const MIN_SCORE = 0.8;

    protected $generator;

    protected $estimator;

    protected $metric;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            'red' => new Blob([255, 0, 0], 3.),
            'green' => new Blob([0, 128, 0], 1.),
            'blue' => new Blob([0, 0, 255], 2.),
        ]);

        $this->estimator = new Pipeline([
            new IntervalDiscretizer(6),
        ], new NaiveBayes(1.0), true);

        $this->metric = new F1Score();
    }

    public function test_build_meta_estimator()
    {
        $this->assertInstanceOf(Pipeline::class, $this->estimator);
        $this->assertInstanceOf(Online::class, $this->estimator);
        $this->assertInstanceOf(Wrapper::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Verbose::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);

        $this->assertEquals(Estimator::CLASSIFIER, $this->estimator->type());

        $this->assertContains(DataFrame::CATEGORICAL, $this->estimator->compatibility());
        $this->assertNotContains(DataFrame::CONTINUOUS, $this->estimator->compatibility());
    }

    public function test_train_partial_predict()
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $testing = $this->generator->generate(self::TEST_SIZE);

        $folds = $training->stratifiedFold(3);

        $this->estimator->train($folds[0]);
        $this->estimator->partial($folds[1]);
        $this->estimator->partial($folds[2]);

        $this->assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    public function test_train_with_unlabeled()
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Unlabeled::quick());
    }

    public function test_predict_untrained()
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

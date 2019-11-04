<?php

namespace Rubix\ML\Tests\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Kernels\SVM\Polynomial;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\AnomalyDetectors\OneClassSVM;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class OneClassSVMTest extends TestCase
{
    protected const TRAIN_SIZE = 400;
    protected const TEST_SIZE = 10;
    protected const MIN_SCORE = 0.5;

    protected const RANDOM_SEED = 0;

    protected $generator;

    protected $estimator;

    protected $metric;

    public function setUp()
    {
        $this->generator = new Agglomerate([
            '0' => new Blob([0., 0.], 0.5),
            '1' => new Circle(0., 0., 8., 0.1),
        ], [0.9, 0.1]);

        $this->estimator = new OneClassSVM(0.01, new Polynomial(4, 1e-3), true, 1e-4);

        $this->metric = new FBeta();

        srand(self::RANDOM_SEED);
    }

    public function test_build_detector()
    {
        $this->assertInstanceOf(OneClassSVM::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);

        $this->assertSame(Estimator::ANOMALY_DETECTOR, $this->estimator->type());

        $this->assertNotContains(DataType::CATEGORICAL, $this->estimator->compatibility());
        $this->assertContains(DataType::CONTINUOUS, $this->estimator->compatibility());

        $this->assertFalse($this->estimator->trained());
    }

    public function test_train_predict()
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    public function test_train_incompatible()
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Unlabeled::quick([['bad']]));
    }

    public function test_predict_untrained()
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick([[1.5]]));
    }
}

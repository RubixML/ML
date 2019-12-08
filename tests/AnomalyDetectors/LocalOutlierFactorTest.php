<?php

namespace Rubix\ML\Tests\AnomalyDetectors;

use Rubix\ML\Learner;
use Rubix\ML\Ranking;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\AnomalyDetectors\LocalOutlierFactor;
use PHPUnit\Framework\TestCase;
use RuntimeException;

class LocalOutlierFactorTest extends TestCase
{
    const TRAIN_SIZE = 350;
    const TEST_SIZE = 10;
    const MIN_SCORE = 0.9;

    protected const RANDOM_SEED = 0;

    /**
     * @var \Rubix\ML\Datasets\Generators\Generator
     */
    protected $generator;

    /**
     * @var \Rubix\ML\AnomalyDetectors\LocalOutlierFactor
     */
    protected $estimator;

    /**
     * @var \Rubix\ML\CrossValidation\Metrics\Metric;
     */
    protected $metric;

    public function setUp() : void
    {
        $this->generator = new Agglomerate([
            '0' => new Blob([0., 0.], 0.5),
            '1' => new Circle(0., 0., 8., 0.1),
        ], [0.9, 0.1]);

        $this->estimator = new LocalOutlierFactor(20, 0.1, new KDTree());

        $this->metric = new FBeta();

        srand(self::RANDOM_SEED);
    }

    public function test_build_detector() : void
    {
        $this->assertInstanceOf(LocalOutlierFactor::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Ranking::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);

        $this->assertSame(Estimator::ANOMALY_DETECTOR, $this->estimator->type());

        $this->assertNotContains(DataType::CATEGORICAL, $this->estimator->compatibility());
        $this->assertContains(DataType::CONTINUOUS, $this->estimator->compatibility());

        $this->assertFalse($this->estimator->trained());

        $this->assertEquals(0, $this->estimator->tree()->height());
    }

    public function test_train_predict() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertGreaterThan(0, $this->estimator->tree()->height());

        $this->assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    public function test_predict_untrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

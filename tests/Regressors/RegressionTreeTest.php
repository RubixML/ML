<?php

namespace Rubix\ML\Tests\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Graph\Trees\CART;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Graph\Trees\DecisionTree;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Datasets\Generators\SwissRoll;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class RegressionTreeTest extends TestCase
{
    protected const TRAIN_SIZE = 300;
    protected const TEST_SIZE = 10;
    protected const MIN_SCORE = 0.9;

    protected const RANDOM_SEED = 0;

    protected $generator;

    protected $estimator;

    protected $metric;

    public function setUp()
    {
        $this->generator = new SwissRoll(4., -7., 0., 1., 0.2);

        $this->estimator = new RegressionTree(10, 2, 3, 1e-7);

        $this->metric = new RSquared();

        srand(self::RANDOM_SEED);
    }

    public function test_build_regressor()
    {
        $this->assertInstanceOf(RegressionTree::class, $this->estimator);
        $this->assertInstanceOf(CART::class, $this->estimator);
        $this->assertInstanceOf(DecisionTree::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);

        $this->assertSame(Estimator::REGRESSOR, $this->estimator->type());

        $this->assertContains(DataType::CATEGORICAL, $this->estimator->compatibility());
        $this->assertContains(DataType::CONTINUOUS, $this->estimator->compatibility());

        $this->assertFalse($this->estimator->trained());

        $this->assertEquals(0, $this->estimator->height());
    }

    public function test_train_predict_feature_importances()
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $this->assertGreaterThan(0, $this->estimator->height());

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);

        $importances = $this->estimator->featureImportances();

        $this->assertCount(3, $importances);
        $this->assertEquals(1., array_sum($importances));
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

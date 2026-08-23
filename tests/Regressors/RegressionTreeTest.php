<?php

namespace Rubix\ML\Tests\Regressors;

use Rubix\ML\Learner;
use Rubix\ML\Encoding;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\RanksFeatures;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Graphviz;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Datasets\Generators\Hyperplane;
use Rubix\ML\Transformers\IntervalDiscretizer;
use Rubix\ML\Graph\Nodes\Split;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

/**
 * @group Regressors
 * @covers \Rubix\ML\Regressors\RegressionTree
 */
class RegressionTreeTest extends TestCase
{
    /**
     * The number of samples in the training set.
     *
     * @var int
     */
    protected const TRAIN_SIZE = 512;

    /**
     * The number of samples in the validation set.
     *
     * @var int
     */
    protected const TEST_SIZE = 256;

    /**
     * The minimum validation score required to pass the test.
     *
     * @var float
     */
    protected const MIN_SCORE = 0.9;

    /**
     * Constant used to see the random number generator.
     *
     * @var int
     */
    protected const RANDOM_SEED = 0;

    /**
     * @var Hyperplane
     */
    protected $generator;

    /**
     * @var RegressionTree
     */
    protected $estimator;

    /**
     * @var RSquared
     */
    protected $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Hyperplane([1.0, 5.5, -7, 0.01], 35.0, 1.0);

        $this->estimator = new RegressionTree(30, 5, 1e-7, 3);

        $this->metric = new RSquared();

        srand(self::RANDOM_SEED);
    }

    protected function assertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(RegressionTree::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(RanksFeatures::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    /**
     * @test
     */
    public function badMaxHeight() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RegressionTree(0);
    }

    /**
     * @test
     */
    public function badMaxLeafSize() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RegressionTree(30, 0);
    }

    /**
     * @test
     */
    public function badMinPurityIncrease() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RegressionTree(30, 5, -1.0);
    }

    /**
     * @test
     */
    public function badMaxFeatures() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RegressionTree(30, 5, 1e-7, 0);
    }

    /**
     * @test
     */
    public function badMaxBins() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RegressionTree(30, 5, 1e-7, 3, 1);
    }

    /**
     * @test
     */
    public function type() : void
    {
        $this->assertEquals(EstimatorType::regressor(), $this->estimator->type());
    }

    /**
     * @test
     */
    public function compatibility() : void
    {
        $expected = [
            DataType::categorical(),
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    /**
     * @test
     */
    public function params() : void
    {
        $expected = [
            'max height' => 30,
            'max leaf size' => 5,
            'min purity increase' => 1.0E-7,
            'max features' => 3,
            'max bins' => null,
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    /**
     * @test
     */
    public function trainPredictImportancesContinuous() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $importances = $this->estimator->featureImportances();

        $this->assertIsArray($importances);
        $this->assertCount(4, $importances);
        $this->assertContainsOnly('float', $importances);

        $dot = $this->estimator->exportGraphviz();

        // Graphviz::dotToImage($dot)->saveTo(new Filesystem('test.png'));

        $this->assertInstanceOf(Encoding::class, $dot);
        $this->assertStringStartsWith('digraph Tree {', $dot);

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    /**
     * @test
     */
    public function trainPredictCategorical() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE + self::TEST_SIZE)
            ->apply(new IntervalDiscretizer(5));

        $testing = $training->randomize()->take(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $dot = $this->estimator->exportGraphviz();

        // Graphviz::dotToImage($dot)->saveTo(new Filesystem('test.png'));

        $this->assertInstanceOf(Encoding::class, $dot);
        $this->assertStringStartsWith('digraph Tree {', $dot);

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    /**
     * @test
     */
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }

    /**
     * @test
     */
    public function trainHeightBalance() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $this->assertGreaterThanOrEqual(2, $this->estimator->height());

        $this->assertIsInt($this->estimator->balance());

        foreach ($this->estimator as $node) {
            if ($node instanceof Split) {
                $this->assertNotNull($node->left());
                $this->assertNotNull($node->right());
            }
        }
    }

    /**
     * @test
     */
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick([[0.5, 0.5, 0.5, 0.5]], ['ok']));
    }

    /**
     * @test
     */
    public function predictIncompatible() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $this->expectException(InvalidArgumentException::class);

        $this->estimator->predict(Unlabeled::quick([[0.5, 0.5, 0.5]]));
    }

    /**
     * Train on a pure dataset, exercising the new purity guard that prunes
     * pure children to leaves without calling the split routine.
     *
     * @test
     */
    public function trainPureDataset() : void
    {
        $samples = array_fill(0, self::TRAIN_SIZE, [128.0, 64.0, 32.0, 16.0]);

        $labels = array_fill(0, self::TRAIN_SIZE, 42.0);

        $this->estimator->train(Labeled::quick($samples, $labels));

        $this->assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict(Unlabeled::quick($samples));

        foreach ($predictions as $prediction) {
            $this->assertEqualsWithDelta(42.0, $prediction, 1e-12);
        }
    }
}

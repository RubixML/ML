<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Regressors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Learner;
use Rubix\ML\Persistable;
use Rubix\ML\RanksFeatures;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Datasets\Generators\Hyperplane;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\IntervalDiscretizer;
use Rubix\ML\Graph\Nodes\Outcome;
use Rubix\ML\Graph\Nodes\Split;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;
use PHPUnit\Framework\TestCase;

#[Group('Regressors')]
#[CoversClass(RegressionTree::class)]
class RegressionTreeTest extends TestCase
{
    /**
     * The number of samples in the training set.
     */
    protected const int TRAIN_SIZE = 512;

    /**
     * The number of samples in the validation set.
     */
    protected const int TEST_SIZE = 256;

    /**
     * The minimum validation score required to pass the test.
     */
    protected const float MIN_SCORE = 0.9;

    /**
     * Constant used to see the random number generator.
     */
    protected const int RANDOM_SEED = 0;

    protected Hyperplane $generator;

    protected RegressionTree $estimator;

    protected RSquared $metric;

    /**
     * @return Generator<string, array{0: int, 1: int}>
     */
    public static function trainedModelCases() : Generator
    {
        yield 'standard split' => [512, 256];

        yield 'smaller split' => [128, 64];
    }

    protected function setUp() : void
    {
        $this->generator = new Hyperplane(
            coefficients: [1.0, 5.5, -7, 0.01],
            intercept: 35.0,
            noise: 1.0
        );

        $this->estimator = new RegressionTree(
            maxHeight: 30,
            maxLeafSize: 5,
            minPurityIncrease: 1e-7,
            maxFeatures: 3
        );

        $this->metric = new RSquared();

        srand(self::RANDOM_SEED);
    }

    #[Test]
    public function preConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(RegressionTree::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(RanksFeatures::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    #[Test]
    public function badMaxHeight() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RegressionTree(maxHeight: 0);
    }

    #[Test]
    public function badMaxLeafSize() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RegressionTree(30, 0);
    }

    #[Test]
    public function badMinPurityIncrease() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RegressionTree(30, 5, -1.0);
    }

    #[Test]
    public function badMaxFeatures() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RegressionTree(30, 5, 1e-7, 0);
    }

    #[Test]
    public function badMaxBins() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RegressionTree(30, 5, 1e-7, 3, 1);
    }

    #[Test]
    public function type() : void
    {
        $this->assertEquals(EstimatorType::regressor(), $this->estimator->type());
    }

    #[Test]
    public function compatibility() : void
    {
        $expected = [
            DataType::categorical(),
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    #[Test]
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

    #[Test]
    public function trainPredictImportancesContinuous() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $importances = $this->estimator->featureImportances();

        $this->assertCount(4, $importances);
        $this->assertContainsOnlyFloat($importances);

        $dot = $this->estimator->exportGraphviz();

        // Graphviz::dotToImage($dot)->saveTo(new Filesystem('test.png'));

        $this->assertStringStartsWith('digraph Tree {', (string) $dot);

        $predictions = $this->estimator->predict($testing);

        /** @var list<float|int> $labels */
        $labels = $testing->labels();
        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    public function trainPredictCategorical() : void
    {
        $training = $this->generator
            ->generate(self::TRAIN_SIZE + self::TEST_SIZE)
            ->apply(new IntervalDiscretizer(bins: 5));

        $testing = $training->randomize()->take(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $dot = $this->estimator->exportGraphviz();

        // Graphviz::dotToImage($dot)->saveTo(new Filesystem('test.png'));

        $this->assertStringStartsWith('digraph Tree {', (string) $dot);

        $predictions = $this->estimator->predict($testing);

        /** @var list<float|int> $labels */
        $labels = $testing->labels();
        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[DataProvider('trainedModelCases')]
    #[Test]
    public function trainedModelExposesAdditionalChecks(int $trainingSize, int $testingSize) : void
    {
        $training = $this->generator->generate($trainingSize);
        $testing = $this->generator->generate($testingSize);

        $this->estimator->train($training);

        self::assertTrue($this->estimator->trained());

        $importances = $this->estimator->featureImportances();

        self::assertCount(4, $importances);
        self::assertContainsOnlyFloat($importances);

        $predictions = $this->estimator->predict($testing);

        self::assertCount($testingSize, $predictions);
    }

    #[Test]
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }

    #[Test]
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

    #[Test]
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick([[0.5, 0.5, 0.5, 0.5]], ['ok']));
    }

    #[Test]
    public function predictIncompatible() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $this->expectException(InvalidArgumentException::class);

        $this->estimator->predict(Unlabeled::quick([[0.5, 0.5, 0.5]]));
    }

    /**
     * Train on two distinct constant feature groups with a constant label, so that
     * the root split produces two non-empty but pure subsets that must be
     * terminated by the purity guard rather than further splitting.
     */
    #[Test]
    public function trainPureChildren() : void
    {
        $groupA = (new Blob([32.0, 32.0, 0.0, 0.0], 0.0))->generate(self::TRAIN_SIZE / 2);

        $groupB = (new Blob([128.0, 128.0, 128.0, 128.0], 0.0))->generate(self::TRAIN_SIZE / 2);

        $training = Labeled::quick(
            array_merge($groupA->samples(), $groupB->samples()),
            array_fill(0, self::TRAIN_SIZE, 42.0)
        );

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict($training);

        foreach ($predictions as $prediction) {
            $this->assertEqualsWithDelta(42.0, $prediction, 1e-12);
        }

        $splitCount = 0;

        foreach ($this->estimator as $node) {
            if ($node instanceof Split) {
                ++$splitCount;

                $this->assertNotSame($node->left(), $node->right());
            } elseif ($node instanceof Outcome) {
                $this->assertEqualsWithDelta(0.0, $node->impurity(), 1e-9);
            }
        }

        $this->assertSame(1, $splitCount);
    }
}

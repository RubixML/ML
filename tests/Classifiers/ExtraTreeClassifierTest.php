<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Classifiers;

use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use Rubix\ML\DataType;
use Rubix\ML\Estimator;
use Rubix\ML\EstimatorType;
use Rubix\ML\Learner;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\RanksFeatures;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Classifiers\ExtraTreeClassifier;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Transformers\IntervalDiscretizer;
use Rubix\ML\Graph\Nodes\Outcome;
use Rubix\ML\Graph\Nodes\Split;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

use function Rubix\ML\argmax;

/**
 * @group Classifiers
 * @covers \Rubix\ML\Classifiers\ExtraTreeClassifier
 */
class ExtraTreeClassifierTest extends TestCase
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

    protected Agglomerate $generator;

    protected ExtraTreeClassifier $estimator;

    protected FBeta $metric;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'red' => new Blob(
                    center: [255, 32, 0],
                    stdDev: 50.0
                ),
                'green' => new Blob(
                    center: [0, 128, 0],
                    stdDev: 10.0
                ),
                'blue' => new Blob(
                    center: [0, 32, 255],
                    stdDev: 30.0
                ),
            ],
            weights: [0.5, 0.2, 0.3]
        );

        $this->estimator = new ExtraTreeClassifier(
            maxHeight: 30,
            maxLeafSize: 16,
            minPurityIncrease: 1e-7,
            maxFeatures: 3
        );

        $this->metric = new FBeta();

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
        $this->assertInstanceOf(ExtraTreeClassifier::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(RanksFeatures::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
    }

    #[Test]
    public function badMaxHeight() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new ExtraTreeClassifier(maxHeight: 0);
    }

    #[Test]
    public function badMaxLeafSize() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new ExtraTreeClassifier(30, 0);
    }

    #[Test]
    public function badMinPurityIncrease() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new ExtraTreeClassifier(30, 16, -1.0);
    }

    #[Test]
    public function badMaxFeatures() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new ExtraTreeClassifier(30, 16, 1e-7, 0);
    }

    #[Test]
    public function type() : void
    {
        $this->assertEquals(EstimatorType::classifier(), $this->estimator->type());
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
            'max leaf size' => 16,
            'min purity increase' => 1.0E-7,
            'max features' => 3,
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

        $this->assertIsArray($importances);
        $this->assertCount(3, $importances);
        $this->assertContainsOnlyFloat($importances);

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $testing->labels()
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    public function trainPredictCategorical() : void
    {
        $training = $this->generator
            ->generate(self::TRAIN_SIZE + self::TEST_SIZE)
            ->apply(new IntervalDiscretizer(3));

        $testing = $training->randomize()->take(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $testing->labels()
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    public function trainPredictProba() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $probabilities = $this->estimator->proba($testing);

        $this->assertIsArray($probabilities);
        $this->assertCount(self::TEST_SIZE, $probabilities);

        $labels = $testing->labels();

        $correct = 0;

        foreach ($probabilities as $offset => $classProbabilities) {
            $this->assertIsArray($classProbabilities);

            $sum = 0.0;

            foreach ($classProbabilities as $probability) {
                $this->assertIsNumeric($probability);
                $this->assertGreaterThanOrEqual(0.0, $probability);
                $this->assertLessThanOrEqual(1.0, $probability);

                $sum += $probability;
            }

            $this->assertEqualsWithDelta(1.0, $sum, 1e-9);

            if (argmax($classProbabilities) === $labels[$offset]) {
                ++$correct;
            }
        }

        $score = $correct / self::TEST_SIZE;

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
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

        $this->estimator->train(Labeled::quick([[0.5, 0.5, 0.5]], [1.0]));
    }

    #[Test]
    public function predictIncompatible() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);

        $this->estimator->train($training);

        $this->expectException(InvalidArgumentException::class);

        $this->estimator->predict(Unlabeled::quick([[0.5, 0.5]]));
    }

    #[Test]
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }

    /**
     * Train on two distinct constant feature groups with different labels, so that
     * the root split produces two non-empty but pure subsets that must be
     * terminated by the purity guard rather than further splitting.
     */
    #[Test]
    public function trainPureChildren() : void
    {
        $training = (new Agglomerate([
            'red' => new Blob([32.0, 32.0, 0.0], 0.0),
            'green' => new Blob([128.0, 128.0, 128.0], 0.0),
        ], [0.5, 0.5]))->generate(self::TRAIN_SIZE);

        $this->assertNotSame($training->label(0), $training->label(self::TRAIN_SIZE / 2));

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $predictions = $this->estimator->predict($training);

        $this->assertEquals($training->labels(), $predictions);

        $splitCount = 0;

        foreach ($this->estimator as $node) {
            if ($node instanceof Split) {
                ++$splitCount;

                $this->assertNotSame($node->left(), $node->right());
            } elseif ($node instanceof Outcome) {
                $this->assertLessThan(1e-9, $node->impurity());
            }
        }

        $this->assertSame(1, $splitCount);
    }

    #[Test]
    public function restoreStateFromSerializedModel() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $restored = unserialize(serialize($this->estimator));

        $this->assertTrue($restored->trained());

        $this->assertEquals($this->estimator->predict($testing), $restored->predict($testing));
    }
}

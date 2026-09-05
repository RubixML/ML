<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Classifiers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\AdaBoost;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

use function Rubix\ML\argmax;

#[Group('Classifiers')]
#[CoversClass(AdaBoost::class)]
class AdaBoostTest extends TestCase
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

    protected AdaBoost $estimator;

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

        $this->estimator = new AdaBoost(
            base: new ClassificationTree(1),
            rate: 1.0,
            ratio: 0.5,
            epochs: 100,
            minChange: 1e-4,
            evalInterval: 3,
            window: 5,
            holdOut: 0.1,
            metric: new FBeta()
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
    public function badLearningRate() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new AdaBoost(base: null, rate: -1e-3);
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
            'base' => new ClassificationTree(1),
            'rate' => 1.0,
            'ratio' => 0.5,
            'epochs' => 100,
            'min change' => 0.0001,
            'eval interval' => 3,
            'window' => 5,
            'hold out' => 0.1,
            'metric' => new FBeta(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    public function trainPredict() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $this->assertTrue($this->estimator->trained());

        $losses = $this->estimator->losses();

        $this->assertIsArray($losses);
        $this->assertContainsOnlyFloat($losses);

        $scores = $this->estimator->scores();

        $this->assertIsArray($scores);
        $this->assertContainsOnlyFloat($scores);

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
        $this->estimator->setLogger(new BlackHole());

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

    /**
     * @test
     */
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
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

    #[Test]
    public function probaRowsSumToOne() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $this->estimator->train($training);

        $probabilities = $this->estimator->proba($testing);

        $this->assertIsArray($probabilities);
        $this->assertCount(self::TEST_SIZE, $probabilities);

        foreach ($probabilities as $probability) {
            $this->assertEqualsWithDelta(1.0, array_sum($probability), 1e-8);
        }
    }
}

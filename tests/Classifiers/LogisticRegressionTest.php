<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Classifiers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\NeuralNet\CostFunctions\BinaryCrossEntropy;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

use function sys_get_temp_dir;
use function uniqid;

#[Group('Classifiers')]
#[CoversClass(LogisticRegression::class)]
class LogisticRegressionTest extends TestCase
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

    protected LogisticRegression $estimator;

    protected FBeta $metric;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'male' => new Blob(
                    center: [69.2, 195.7, 40.0],
                    stdDev: [2.0, 6.0, 0.6]
                ),
                'female' => new Blob(
                    center: [63.7, 168.5, 38.1],
                    stdDev: [1.6, 5.0, 0.8]
                ),
            ],
            weights: [0.45, 0.55]
        );

        $this->estimator = new LogisticRegression(
            batchSize: 100,
            optimizer: new Adam(rate: 0.01),
            l2Penalty: 1e-4,
            epochs: 300,
            minChange: 1e-4,
            evalInterval: 3,
            window: 5,
            holdOut: 0.1,
            costFn: new BinaryCrossEntropy(),
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
    public function badBatchSize() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new LogisticRegression(batchSize: -100);
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
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    #[Test]
    public function params() : void
    {
        $expected = [
            'batch size' => 100,
            'optimizer' => new Adam(0.01),
            'l2 penalty' => 1e-4,
            'epochs' => 300,
            'min change' => 1e-4,
            'eval interval' => 3,
            'window' => 5,
            'hold out' => 0.1,
            'cost fn' => new BinaryCrossEntropy(),
            'metric' => new FBeta(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    #[Test]
    public function trainPartialPredict() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $dataset = $this->generator->generate(self::TRAIN_SIZE + self::TEST_SIZE);

        $dataset->apply(new ZScaleStandardizer());

        $testing = $dataset->randomize()->take(self::TEST_SIZE);

        $folds = $dataset->stratifiedFold(3);

        $this->estimator->train($folds[0]);
        $this->estimator->partial($folds[1]);
        $this->estimator->partial($folds[2]);

        $this->assertTrue($this->estimator->trained());

        $losses = $this->estimator->losses();

        $this->assertIsArray($losses);
        $this->assertContainsOnlyFloat($losses);

        $scores = $this->estimator->scores();

        $this->assertIsArray($scores);
        $this->assertContainsOnlyFloat($scores);

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
    public function snapshotPathIsTransientAndResolvedLazily() : void
    {
        $this->estimator->setLogger(new BlackHole());

        $dataset = $this->generator->generate(self::TRAIN_SIZE + self::TEST_SIZE);

        $dataset->apply(new ZScaleStandardizer());

        $snapshotPath = sys_get_temp_dir() . '/rubix-ml-test-' . uniqid() . '.dat';

        $this->estimator->setSnapshotPath($snapshotPath);

        $this->estimator->train($dataset->stratifiedFold(2)[0]);

        $this->assertTrue($this->estimator->trained());

        $this->assertArrayNotHasKey('snapshotPath', $this->estimator->__serialize());

        $copy = unserialize(serialize($this->estimator));

        $this->assertTrue($copy->trained());

        $this->assertArrayNotHasKey('snapshotPath', $copy->__serialize());

        $copy->partial($dataset->stratifiedFold(2)[0]);

        $this->assertArrayNotHasKey('snapshotPath', $copy->__serialize());
    }

    #[Test]
    public function snapshotPathRejectsDirectory() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->setSnapshotPath(sys_get_temp_dir());
    }

    #[Test]
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick(samples: [['bad']], labels: ['green']));
    }

    #[Test]
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

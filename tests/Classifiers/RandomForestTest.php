<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Classifiers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Tests\DataProvider\BackendProviderTrait;

#[Group('Classifiers')]
#[CoversClass(RandomForest::class)]
class RandomForestTest extends TestCase
{
    use BackendProviderTrait;

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

    protected RandomForest $estimator;

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

        $this->estimator = new RandomForest(
            base: new ClassificationTree(maxHeight: 3),
            estimators: 50,
            ratio: 0.2,
            balanced: true
        );

        $this->metric = new FBeta();

        srand(self::RANDOM_SEED);
    }

    public function testAssertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    public function testBadNumEstimators() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new RandomForest(base: null, estimators: -100);
    }

    public function testType() : void
    {
        $this->assertEquals(EstimatorType::classifier(), $this->estimator->type());
    }

    public function testCompatibility() : void
    {
        $expected = [
            DataType::categorical(),
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    public function testParams() : void
    {
        $expected = [
            'base' => new ClassificationTree(3),
            'estimators' => 50,
            'ratio' => 0.2,
            'balanced' => true,
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    #[DataProvider('provideBackends')]
    public function testTrainPredictImportances(Backend $backend) : void
    {
        $this->estimator->setBackend($backend);

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

    public function testPredictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

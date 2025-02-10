<?php

declare(strict_types=1);

namespace Rubix\ML\Tests;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Pipeline;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\Transformers\PolynomialExpander;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('MetaEstimators')]
#[CoversClass(Pipeline::class)]
class PipelineTest extends TestCase
{
    protected const int TRAIN_SIZE = 512;

    protected const int TEST_SIZE = 256;

    protected const float MIN_SCORE = 0.8;

    protected const int RANDOM_SEED = 0;

    protected Agglomerate $generator;

    protected Pipeline $estimator;

    protected Accuracy $metric;

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

        $this->estimator = new Pipeline(
            transformers: [
                new PolynomialExpander(2),
                new ZScaleStandardizer(),
            ],
            base: new SoftmaxClassifier(),
            elastic: true
        );

        $this->metric = new Accuracy();

        srand(self::RANDOM_SEED);
    }

    public function testType() : void
    {
        $this->assertEquals(EstimatorType::classifier(), $this->estimator->type());
    }

    public function testCompatibility() : void
    {
        $expected = [
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->estimator->compatibility());
    }

    public function testParams() : void
    {
        $expected = [
            'transformers' => [
                new PolynomialExpander(2),
                new ZScaleStandardizer(),
            ],
            'estimator' => new SoftmaxClassifier(),
            'elastic' => true,
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    public function testTrainPartialPredict() : void
    {
        $training = $this->generator->generate(self::TRAIN_SIZE);
        $testing = $this->generator->generate(self::TEST_SIZE);

        $folds = $training->stratifiedFold(3);

        $this->estimator->train($folds[0]);
        $this->estimator->partial($folds[1]);
        $this->estimator->partial($folds[2]);

        $this->assertTrue($this->estimator->trained());

        /** @var list<int|string> $predictions */
        $predictions = $this->estimator->predict($testing);

        /** @var list<int|string> $labels */
        $labels = $testing->labels();

        $score = $this->metric->score(
            predictions: $predictions,
            labels: $labels
        );

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    public function testPredictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }

    protected function testAssertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }
}

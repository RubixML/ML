<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Classifiers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\Encoding;
use Rubix\ML\EstimatorType;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Noise;
use Rubix\ML\NeuralNet\Layers\Dropout;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Datasets\Generators\Circle;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Classifiers')]
#[CoversClass(MultilayerPerceptron::class)]
class MultilayerPerceptronTest extends TestCase
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

    protected MultilayerPerceptron $estimator;

    protected FBeta $metric;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'inner' => new Circle(
                    x: 0.0,
                    y: 0.0,
                    scale: 1.0,
                    noise: 0.01
                ),
                'middle' => new Circle(
                    x: 0.0,
                    y: 0.0,
                    scale: 5.0,
                    noise: 0.05
                ),
                'outer' => new Circle(
                    x: 0.0,
                    y: 0.0,
                    scale: 10.0,
                    noise: 0.1
                ),
            ],
            weights: [3, 3, 4]
        );

        $this->estimator = new MultilayerPerceptron(
            hiddenLayers: [
                new Dense(32),
                new Activation(new LeakyReLU(0.1)),
                new Dropout(0.1),
                new Dense(16),
                new Activation(new LeakyReLU(0.1)),
                new Noise(1e-5),
                new Dense(8),
                new Activation(new LeakyReLU(0.1)),
            ],
            batchSize: 32,
            optimizer: new Adam(rate: 0.001),
            epochs: 100,
            minChange: 1e-3,
            evalInterval: 3,
            window: 5,
            holdOut: 0.1,
            costFn: new CrossEntropy(),
            metric: new FBeta()
        );

        $this->metric = new FBeta();

        srand(self::RANDOM_SEED);
    }

    public function testAssertPreConditions() : void
    {
        $this->assertFalse($this->estimator->trained());
    }

    public function testBadBatchSize() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new MultilayerPerceptron(hiddenLayers: [], batchSize: -100);
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
            'hidden layers' => [
                new Dense(32),
                new Activation(new LeakyReLU(0.1)),
                new Dropout(0.1),
                new Dense(16),
                new Activation(new LeakyReLU(0.1)),
                new Noise(1e-5),
                new Dense(8),
                new Activation(new LeakyReLU(0.1)),
            ],
            'batch size' => 32,
            'optimizer' => new Adam(0.001),
            'epochs' => 100,
            'min change' => 1e-3,
            'eval interval' => 3,
            'window' => 5,
            'hold out' => 0.1,
            'cost fn' => new CrossEntropy(),
            'metric' => new FBeta(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    public function testTrainPartialPredict() : void
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

        $dot = $this->estimator->exportGraphviz();

        // Graphviz::dotToImage($dot)->saveTo(new Filesystem('test.png'));

        $this->assertInstanceOf(Encoding::class, $dot);
        $this->assertStringStartsWith('digraph Tree {', (string) $dot);

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

    public function testTrainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick(samples: [['bad']], labels: ['green']));
    }

    public function testPredictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}

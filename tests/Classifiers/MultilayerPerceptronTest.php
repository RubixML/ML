<?php

namespace Rubix\ML\Tests\Classifiers;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\DataType;
use Rubix\ML\Encoding;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Probabilistic;
use Rubix\ML\EstimatorType;
use Rubix\ML\Helpers\Graphviz;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Persisters\Filesystem;
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

/**
 * @group Classifiers
 * @covers \Rubix\ML\Classifiers\MulitlayerPerceptron
 */
class MultilayerPerceptronTest extends TestCase
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
     * @var Agglomerate
     */
    protected $generator;

    /**
     * @var MultilayerPerceptron
     */
    protected $estimator;

    /**
     * @var FBeta
     */
    protected $metric;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->generator = new Agglomerate([
            'inner' => new Circle(0.0, 0.0, 1.0, 0.01),
            'middle' => new Circle(0.0, 0.0, 5.0, 0.05),
            'outer' => new Circle(0.0, 0.0, 10.0, 0.1),
        ], [3, 3, 4]);

        $this->estimator = new MultilayerPerceptron([
            new Dense(32),
            new Activation(new LeakyReLU(0.1)),
            new Dropout(0.1),
            new Dense(16),
            new Activation(new LeakyReLU(0.1)),
            new Noise(1e-5),
            new Dense(8),
            new Activation(new LeakyReLU(0.1)),
        ], 32, new Adam(0.001), 1e-4, 100, 1e-3, 5, 0.1, new CrossEntropy(), new FBeta());

        $this->metric = new FBeta();

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
        $this->assertInstanceOf(MultilayerPerceptron::class, $this->estimator);
        $this->assertInstanceOf(Online::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Probabilistic::class, $this->estimator);
        $this->assertInstanceOf(Verbose::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    /**
     * @test
     */
    public function badBatchSize() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new MultilayerPerceptron([], -100);
    }

    /**
     * @test
     */
    public function type() : void
    {
        $this->assertEquals(EstimatorType::classifier(), $this->estimator->type());
    }

    /**
     * @test
     */
    public function compatibility() : void
    {
        $expected = [
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
            'l2 penalty' => 1e-4,
            'epochs' => 100,
            'min change' => 1e-3,
            'window' => 5,
            'hold out' => 0.1,
            'cost fn' => new CrossEntropy(),
            'metric' => new FBeta(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }

    /**
     * @test
     */
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

        $dot = $this->estimator->exportGraphviz();

        // Graphviz::dotToImage($dot)->saveTo(new Filesystem('test.png'));

        $this->assertInstanceOf(Encoding::class, $dot);
        $this->assertStringStartsWith('digraph Tree {', $dot);

        $losses = $this->estimator->losses();

        $this->assertIsArray($losses);
        $this->assertContainsOnly('float', $losses);

        $scores = $this->estimator->scores();

        $this->assertIsArray($scores);
        $this->assertContainsOnly('float', $scores);

        $predictions = $this->estimator->predict($testing);

        $score = $this->metric->score($predictions, $testing->labels());

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    /**
     * @test
     */
    public function trainIncompatible() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Labeled::quick([['bad']], ['green']));
    }

    /**
     * @test
     */
    public function predictUntrained() : void
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}
